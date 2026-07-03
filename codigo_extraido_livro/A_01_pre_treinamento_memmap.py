import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data  import Dataset , DataLoader
from torch.cuda.amp import  autocast , GradScaler

# 1. Dataset  baseado em NumPy Memmap para  carregar tokens  diretamente do disco
class MemmapDataset(Dataset):
    def __init__(self , bin_path: str , seq_len: int = 1024):
        self.bin_path = bin_path
        self.seq_len = seq_len
        # O arquivo binário contém tokens   uint16 salvos consecutivamente
        self.tokens = np.memmap(bin_path , dtype=np.uint16 , mode='r')
        self.num_samples =  len(self.tokens) // seq_len

    def __len__(self) ->  int:
        return  self.num_samples

    def __getitem__(self , idx: int) -> tuple [torch.Tensor , torch.Tensor ]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        # Ler seq_len + 1 tokens  para query e target causal
        chunk = torch.from_numpy(self.tokens[start: end]. astype(np.int64))
        x = chunk [:-1]
        y = chunk [1:]
        return x, y

# 2. Scheduler  Cossenoide com Warmup Linear
class CosineWarmupScheduler :
    def __init__(self , optimizer , lr_max: float , steps_warmup: int , steps_total: int , lr_min: float = 3e-5):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.steps_warmup = steps_warmup
        self.steps_total = steps_total
        self.lr_min = lr_min

    def step(self , step: int):
        if step < self.steps_warmup:
            # Warmup  linear
            lr = self.lr_max * (step / self.steps_warmup)
        else:
            # Decaimento  cosseno
            progress = (step - self.steps_warmup) / (self.steps_total - self.steps_warmup)
            progress = min (1.0, max (0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos( math.pi * progress))
            lr = self.lr_min + (self.lr_max - self.lr_min) * cosine_decay

        for  param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# 3. Loop de Treinamento  Autoregressivo  Principal
def treinar_minilm (modelo: nn.Module ,
                    dataset_path: str ,
                    batch_size: int = 8,
                    acumulacao_grad : int = 8,
                    max_passos: int = 10000 ,
                    checkpoint_dir : str = "./ checkpoints"):
    os.makedirs(checkpoint_dir , exist_ok=True)
    device = torch.device( "cuda" if torch.cuda.is_available () else "cpu")
    modelo = modelo.to(device)

    # Dataset e DataLoader  assincrono
    dataset = MemmapDataset(dataset_path , seq_len =1024)
    dataloader = DataLoader(dataset , batch_size= batch_size , shuffle=True , pin_memory=True , drop_last=True)

    # Divisão de decaimento de peso (excluir   biases e normalizacoes)
    param_dict = {pn: p  for pn , p in modelo.named_parameters () if p.requires_grad}
    decay_params = [p  for n, p in param_dict.items () if p.dim()  >= 2]
    nodecay_params = [p  for n, p in param_dict.items () if p.dim() < 2]

    optim_groups = [
        {"params" : decay_params , "weight_decay": 0.1},
        {"params" : nodecay_params , "weight_decay": 0.0}
    ]

    otimizador = torch.optim.AdamW(optim_groups , lr =3e-4, betas =(0.9 , 0.95) , eps=1e-8)
    scheduler = CosineWarmupScheduler (otimizador , lr_max =3e-4, steps_warmup =1000 ,steps_total =max_passos)

    # Para treino  estavel em precisao mista
    scaler = GradScaler ()

    passo = 0
    otimizador.zero_grad ()

    while passo < max_passos:
        for x, y in  dataloader:
            x, y = x.to(device), y.to(device)

            # Forward sob  autocast AMP (Mixed Precision)
            with  autocast(dtype=torch.bfloat16):
                logits = modelo(x)
                # Reshape para calcular a entropia cruzada
                loss = nn.functional.cross_entropy( logits.view(-1, logits.size (-1)), y.view (-1))
                loss = loss / acumulacao_grad   # Escalonamento por  acumulação

            # Backward
            scaler.scale(loss).backward ()

            if (passo + 1) % acumulacao_grad == 0:
                # Gradient Clipping
                scaler.unscale_(otimizador)
                nn.utils.clip_grad_norm_ (modelo.parameters (), max_norm =1.0)

                # Passo do otimizador e atualizacao da escala
                scaler.step(otimizador)
                scaler.update ()
                otimizador.zero_grad ()

                lr_atual = scheduler.step(passo)

                if (passo // acumulacao_grad ) % 100 == 0:
                    print(f"Passo: {passo // acumulacao_grad } | Loss: { loss.item () * acumulacao_grad :.4f} | LR: {lr_atual :.6f}")

            # Salvar  checkpoint
            if (passo + 1) % 5000 == 0:
                caminho_check = os.path.join( checkpoint_dir , f" checkpoint_step_ {passo }.pt")
                torch.save ({
                    'step ': passo ,
                    'model_state_dict ': modelo.state_dict (),
                    'optimizer_state_dict ': otimizador.state_dict (),
                    'loss ': loss.item (),
                }, caminho_check)

            passo += 1
            if passo  >= max_passos:
                break
