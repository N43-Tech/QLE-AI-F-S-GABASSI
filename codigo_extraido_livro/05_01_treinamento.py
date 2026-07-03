import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim  import AdamW
from torch.optim.lr_scheduler  import LambdaLR
from torch.cuda.amp import  GradScaler , autocast
from torch.utils.data  import DataLoader , Dataset
import numpy as np
from pathlib  import Path
import math
import time
import json
from typing  import Optional , Dict

# =======================================================
# 1) Dataset de tokens pré-processados (memmap biná rio)
# =======================================================

class DatasetTokens(Dataset):
    """
    Dataset  eficiente usando numpy memmap  para grandes corpora.
    Os tokens são armazenados  como array binário de uint16.

    Parâmetros:
        caminho_dados: arquivo .bin com tokens uint16
        comprimento_bloco : tamanho de cada sequê ncia de treinamento
    """

    def __init__(self , caminho_dados: str , comprimento_bloco : int = 1024):
        self. comprimento_bloco = comprimento_bloco

        dados = np.memmap(caminho_dados , dtype=np.uint16 , mode='r')
        self.dados = dados
        self.n_blocos =  len(dados) // ( comprimento_bloco + 1)

        print(f"Dataset: {len(dados):,} tokens , "
                f"{self.n_blocos :,} blocos de { comprimento_bloco +1}")

    def __len__(self) ->  int:
        return  self.n_blocos

    def __getitem__(self , idx: int) -> tuple :
        # Pegar  bloco de comprimento_bloco +1 tokens
        inicio = idx * (self. comprimento_bloco + 1)
        trecho = self.dados[inicio : inicio + self.comprimento_bloco + 1]

        x = torch.from_numpy(trecho [: -1]. astype(np.int64))
        y = torch.from_numpy(trecho [1:]. astype(np.int64))
        return x, y


# =======================================================
# 2) Scheduler: warmup + cosseno
# =======================================================

def criar_scheduler_cossenoide (
    otimizador ,
    eta_max: float ,
    eta_min: float ,
    T_warmup: int ,
    T_total: int
) -> LambdaLR:
    """
    Cria scheduler de taxa de aprendizado   com warmup linear + cosseno.

    Parâmetros:
        otimizador: instância do otimizador   PyTorch
        eta_max: taxa de aprendizado máxima
        eta_min: taxa mínima (após decaimento completo)
        T_warmup: número de passos de warmup
        T_total: número  total de passos de treinamento
    """
    def _lr_lambda(passo:  int) -> float:
        if passo < T_warmup:
            # Warmup  linear: 0 -> 1.0
            return passo / max(1, T_warmup)
        else:
            # Decaimento  cosseno
            progresso = (passo - T_warmup) / max (1, T_total - T_warmup)
            cosseno = 0.5 * (1.0 + math.cos(math.pi * progresso))
            # Escalar  entre eta_min/eta_max e 1.0
            fator_min = eta_min / eta_max
            return fator_min + (1.0 - fator_min) * cosseno

    return LambdaLR(otimizador , lr_lambda= _lr_lambda)


# =======================================================
# 3) Função de perda com  ignorar padding
# =======================================================

def calcular_perda (logits: torch.Tensor ,
                    alvos: torch.Tensor ,
                    id_padding: int = 0) -> torch.Tensor:
    """
    Entropia  cruzada ignorando tokens de padding.

    Parâmetros:
        logits: (batch , seq , vocab)
        alvos:   (batch , seq)
        id_padding: ID do token de padding ( ignorado no loss)
    """
    B, T, V = logits.shape
    logits_planos = logits.view(B * T, V)
    alvos_planos   = alvos.view(B * T)

    return F.cross_entropy(
        logits_planos , alvos_planos ,
        ignore_index=id_padding
    )


# =======================================================
# 4) Loop de treinamento  principal
# =======================================================

class TreinadorLLM:
    """
    Treinador  completo para o MiniLM -PT.

    Características:
        - Precisão mista  BF16/FP32 automática
        - Gradient  accumulation para simular batches maiores
        - Checkpoint  automático
        - Logging  detalhado de métricas

    Parâmetros:
        modelo: instância do MiniLMPT
        cfg_treino: dicionário de configuração de treinamento
        dispositivo: 'cuda ', 'mps' ou 'cpu'
    """

    def __init__(self ,
                modelo: nn.Module ,
                cfg_treino: dict ,
                dispositivo: str = 'auto '):
        self.modelo = modelo
        self.cfg = cfg_treino

        if dispositivo ==  'auto ':
            if torch.cuda.is_available ():
                self.dispositivo = torch.device( ' cuda ')
            elif  torch.backends.mps.is_available ():
                self.dispositivo = torch.device( ' mps')
            else :
                self.dispositivo = torch.device( ' cpu')
        else:
            self.dispositivo = torch.device( dispositivo)

        print(f"Dispositivo de treinamento: {self.dispositivo}")

        self.modelo = self.modelo.to(self.dispositivo)

        # Compilar  modelo (PyTorch 2.0+)  para speedup
        if hasattr (torch , 'compile ') and self.dispositivo.type == 'cuda ':
            print("Compilando modelo com torch.compile ()...")
            self.modelo = torch. compile(self.modelo)

        # Otimizador  AdamW
        # Separar parâmetros com e sem weight   decay
        decaimento , sem_decaimento = [], []
        for nome , param in modelo. named_parameters ():
            if param.dim()  >= 2:
                decaimento.append(param)
            else :
                sem_decaimento .append(param)

        grupos_param = [
            { 'params ': decaimento ,  'weight_decay ': cfg_treino['weight_decay ']},
            { 'params ': sem_decaimento , ' weight_decay ': 0.0}
        ]

        self.otimizador = AdamW(
            grupos_param ,
            lr=cfg_treino[ 'eta_max '],
            betas =( cfg_treino['beta1 '], cfg_treino[ 'beta2 ']),
            eps=cfg_treino[ 'epsilon ']
        )

        # Scheduler
        self.scheduler = criar_scheduler_cossenoide (
            self.otimizador ,
            eta_max=cfg_treino['eta_max '],
            eta_min=cfg_treino['eta_min '],
            T_warmup=cfg_treino['warmup_passos '],
            T_total=cfg_treino['total_passos ']
        )

        # Scaler  para precisão mista
        self.usar_bf16 = (self.dispositivo.  type == 'cuda ' and
                            torch.cuda.is_bf16_supported ())
        self.scaler = GradScaler(enabled=  not self.usar_bf16)

        self.passo_global = 0
        self.historico: Dict[ str , list] = {
            'perda_treino ': [], 'perda_val ': [],
            'perplexidade_val ': [], 'lr': [], ' tokens_por_segundo ': []
        }

    def _passo_treino(self , lote_x: torch.Tensor ,
                        lote_y: torch.Tensor) -> float:
        """ Executa um passo de treinamento e retorna a perda."""
        lote_x = lote_x.to(self.dispositivo)
        lote_y = lote_y.to(self.dispositivo)

        dtype = torch.bfloat16  if self.usar_bf16 else torch.float16

        with  autocast(device_type=self.dispositivo.type ,
                        dtype=dtype ,
                        enabled =( self.dispositivo.type == 'cuda ')):
            logits , _ = self.modelo(lote_x)
            perda = calcular_perda(logits , lote_y)
            perda = perda / self.cfg[' acum_gradiente ']

        if self.usar_bf16:
            perda.backward ()
        else:
            self.scaler.scale(perda).backward ()

        return  perda.item () * self.cfg[' acum_gradiente ']

    def treinar(self ,
                loader_treino: DataLoader ,
                loader_val: Optional[DataLoader] = None ,
                dir_checkpoint : str = './ checkpoints ') -> None:
        """
        Loop de treinamento  principal.

        Parâmetros:
            loader_treino: DataLoader de treinamento
            loader_val: DataLoader de validação ( opcional)
            dir_checkpoint : diretório para salvar checkpoints
        """
        Path(dir_checkpoint ).mkdir(parents=True , exist_ok=True)

        self.modelo.train ()
        acumulados = 0
        perda_acum = 0.0
        t_inicio = time.time ()

        print(f"\n{ '= '*60}")
        print(f"Iniciando treinamento do MiniLM -PT")
        print(f"  Total de passos:   {self.cfg[' total_passos ']:,}")
        print(f"  Warmup passos:     {self.cfg[' warmup_passos ']:,}")
        print(f"  Batch size efetivo: {self.cfg[' batch_size '] * self.cfg[' acum_gradiente ']}")
        print(f"  Precisão:          {'BF16 ' if self.usar_bf16 else 'FP16 '}")
        print(f"{ '= '*60}\n")

        for lote_x , lote_y in loader_treino:
            if self.passo_global  >= self.cfg[ ' total_passos ']:
                break

            # Acumulação de gradiente
            perda_acum += self._passo_treino(lote_x, lote_y)
            acumulados += 1

            if acumulados < self.cfg[ ' acum_gradiente ']:
                continue

            # --- Passo  completo de otimização ---
            acumulados = 0

            # Clip de gradiente
            if self.usar_bf16:
                norm_grad = nn.utils.clip_grad_norm_ (
                    self.modelo.parameters (),
                    self.cfg['clip_gradiente ']
                )
                self.otimizador.step ()
            else :
                self.scaler.unscale_(self.otimizador)
                norm_grad = nn.utils.clip_grad_norm_ (
                    self.modelo.parameters (),
                    self.cfg['clip_gradiente ']
                )
                self.scaler.step(self.otimizador)
                self.scaler.update ()

            self.otimizador.zero_grad(set_to_none= True)
            self.scheduler.step ()
            self.passo_global += 1

            # Logging
            if self.passo_global % self.cfg[  ' log_intervalo '] == 0:
                t_agora = time.time ()
                dt = t_agora - t_inicio
                tokens_s = (self.cfg['batch_size '] * self.cfg['acum_gradiente '] *
                            self.cfg[' comprimento_max '] *
                            self.cfg['log_intervalo '] / dt)
                lr_atual = self.scheduler.get_last_lr ()[0]

                self.historico['perda_treino '].append(perda_acum)
                self.historico['lr']. append( lr_atual)
                self.historico['tokens_por_segundo ']. append(tokens_s)

                print(f"Passo {self.passo_global :6d}/{ self.cfg['total_passos ']} | "
                        f"Perda: {perda_acum :.4f} | "
                        f"PPL: {math.exp(min( perda_acum , 20)):.1f} | "
                        f"LR: {lr_atual :.2e} | "
                        f"|grad |: {norm_grad :.3f} | "
                        f"{tokens_s /1000:.1f}K tok/s")

                perda_acum = 0.0
                t_inicio = t_agora

            # Avaliação
            if (loader_val  is not None and
                self.passo_global % self.cfg[ ' eval_intervalo '] == 0):
                perda_val = self._avaliar( loader_val)
                ppl_val = math.exp(min(perda_val , 20))
                self.historico['perda_val ']. append( perda_val)
                self.historico['perplexidade_val '].append(ppl_val)
                print(f"\n  [VAL] Passo {self.passo_global} | "
                        f"Perda Val: {perda_val :.4f} | PPL Val: {ppl_val :.1f}\n")
                self.modelo.train ()

            # Checkpoint
            if self.passo_global % self.cfg[  ' save_intervalo '] == 0:
                self. _salvar_checkpoint ( dir_checkpoint )

        print( "\nTreinamento concluído!")

    @torch.no_grad ()
    def _avaliar(self , loader_val: DataLoader ,
                    n_lotes: int = 20) -> float :
        """ Calcula perda de validação em n_lotes lotes."""
        self.modelo. eval ()
        perdas = []

        for i, (lote_x , lote_y) in enumerate ( loader_val):
            if i >= n_lotes:
                break
            lote_x = lote_x.to(self.dispositivo)
            lote_y = lote_y.to(self.dispositivo)

            logits , _ = self.modelo(lote_x)
            perda = calcular_perda(logits , lote_y)
            perdas.append(perda.item ())

        return np.mean(perdas)

    def _salvar_checkpoint (self , dir_checkpoint: str ) -> None:
        """ Salva checkpoint do modelo e estado do otimizador."""
        caminho = Path(dir_checkpoint ) / f" minilm_passo_{self.passo_global }.pt"

        # Unwrap do modelo  compilado
        estado_modelo = (self.modelo._orig_mod.state_dict ()
                        if hasattr(self.modelo , ' _orig_mod ')
                        else self.modelo.state_dict ())

        torch.save ({
            'passo ': self.passo_global ,
            'modelo ': estado_modelo ,
            'otimizador ': self.otimizador.state_dict (),
            'scheduler ': self.scheduler.state_dict (),
            'scaler ': self.scaler.state_dict (),
            'historico ': self.historico ,
            'cfg_modelo ': CFG_MINILM ,
            'cfg_treino ': self.cfg ,
        }, caminho)
        print(f"  Checkpoint salvo: {caminho}" )


# Configuração de treinamento
CFG_TREINO = {
    'eta_max ':          3e-4,    # Taxa de aprendizado máxima
    'eta_min ':          3e-5,    # Taxa mínima (10% do max)
    'beta1 ':            0.9,
    'beta2 ':            0.95,
    'epsilon ':          1e-8,
    'weight_decay ':     0.1,
    'clip_gradiente ':   1.0,
    'batch_size ':         16,    # Lotes por passo de acumulação
    'acum_gradiente ':    4,      # Simula  batch de64 sequências
    'comprimento_max ': 1024,
    'total_passos ':  50000 ,     # ~3B tokens  para 85M parâmetros
    'warmup_passos ':  2000,      # 4% dos passos
    'log_intervalo ':    100,
    'eval_intervalo ':  1000,
    'save_intervalo ':  5000,
}

# Tokens totais de treinamento:
# 50000 * 16 * 4 * 1024 = ~3.3 bilhões de tokens
print(f"Tokens  totais de treinamento: "
        f"{CFG_TREINO['total_passos '] * CFG_TREINO[' batch_size '] * CFG_TREINO[' acum_gradiente '] * CFG_TREINO[' comprimento_max '] / 1e9:.2f}B")
