import torch
import torch.nn as nn
import math
from typing  import Optional

class CamadaLoRA(nn.Module):
    """
    Camada Linear com LoRA (Low -Rank Adaptation).

    Pode operar em dois  modos:
        - Treinamento: W_x + (alpha/r) * B * A * x
        - Inferência  merged: (W + delta_W) * x

    Parâmetros:
        dim_entrada: n (dimensão de entrada)
        dim_saida: m (dimensão de saída)
        rank: r (rank da decomposição)
        alpha: fator de escala (tipicamente = rank)
        dropout_lora: dropout  nas ativações LoRA
        bias: se True , inclui bias
    """

    def __init__(self ,
                dim_entrada: int ,
                dim_saida: int ,
                rank: int = 8,
                alpha: float = 16.0,
                dropout_lora: float = 0.05,
                bias: bool = False):
        super ().__init__ ()

        self.dim_entrada = dim_entrada
        self.dim_saida = dim_saida
        self.rank = rank
        self.escala = alpha / rank
        self.merged = False

        # Peso base (congelado)
        self.peso_base = nn.Parameter(
            torch.empty(dim_saida , dim_entrada), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(dim_saida), requires_grad=False
            )
        else:
            self.bias = None

        # Adaptadores  LoRA (treináveis)
        self.lora_A = nn.Parameter(
            torch.empty(rank , dim_entrada)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(dim_saida , rank)
        )
        self.lora_dropout = nn.Dropout(dropout_lora)

        # Inicialização
        nn.init. kaiming_uniform_ (self.peso_base , a= math.sqrt (5))
        nn.init. kaiming_uniform_ (self.lora_A , a= math.sqrt (5))
        # lora_B já é zero: garante  que  delta_W =0 no início

    @property
    def delta_W(self) -> torch.Tensor:
        """ Calcula delta_W = (alpha/r) * B * A."""
        return  self.escala * (self.lora_B @ self.lora_A)

    def merge(self) -> None:
        """
        Mescla os pesos  LoRA ao peso base.
        Após merge , a camada funciona como uma Linear padrão
        sem  overhead de computação adicional.
        """
        if not self.merged:
            self.peso_base.data += self.delta_W
            self.merged = True

    def unmerge(self) -> None:
        """ Desfaz o merge (para continuar  treinando)."""
        if self.merged:
            self.peso_base.data  -= self.delta_W
            self.merged = False

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        """
        Parâmetros:
            x: (...,  dim_entrada)
        Retorna:
            (..., dim_saida)
        """
        if self.merged:
            # Modo  merged: apenas multiplicação pelo peso fundido
            saida = x @ self.peso_base.T
        else:
            # Modo LoRA: W*x + (alpha/r) * B * A * dropout(x)
            saida_base = x @ self.peso_base.T
            x_dropout = self.lora_dropout(x)
            delta_saida = (x_dropout @ self.lora_A.T) @ self.lora_B.T
            saida = saida_base + self.escala * delta_saida

        if self.bias  is not None:
            saida = saida + self.bias

        return  saida


def aplicar_lora_ao_modelo (
    modelo: nn.Module ,
    rank: int = 8,
    alpha: float = 16.0,
    modulos_alvo: list  = ['W_q', 'W_v'],
    dropout_lora: float  = 0.05
) -> nn.Module:
    """
    Substitui  camadas Linear especificadas  por CamadaLoRA.
    Congela  todos os outros parâmetros.

    Parâmetros:
        modelo: LLM  original
        rank: rank LoRA
        alpha: fator de escala
        modulos_alvo: lista de nomes de módulos a adaptar
        dropout_lora: dropout  nas ativações LoRA
    Retorna:
        modelo com LoRA  aplicado
    """
    # Congelar  todos os parâmetros
    for param in modelo.parameters ():
        param. requires_grad_(False)

    n_substituidos = 0

    for nome , modulo in modelo.named_modules ():
        # Verificar se o módulo deve ser   adaptado
        nome_curto = nome.split( '.')[-1]
        if nome_curto  not in modulos_alvo:
            continue
        if not  isinstance(modulo , nn.Linear):
            continue

        # Criar  camada LoRA com mesmas  dimensões
        lora_camada = CamadaLoRA(
            dim_entrada=modulo.in_features ,
            dim_saida=modulo.out_features ,
            rank=rank ,
            alpha=alpha ,
            dropout_lora=dropout_lora ,
            bias=modulo.bias  is not None
        )

        # Copiar  pesos existentes
        lora_camada.peso_base.data = modulo.weight.data.clone ()
        if modulo.bias  is not None:
            lora_camada.bias.data = modulo.bias.data.clone ()

        # Substituir módulo
        parent_name , child_name = nome.rsplit( '.', 1)
        parent = modelo.get_submodule(parent_name)
        setattr (parent , child_name , lora_camada)
        n_substituidos += 1

    # Contar parâmetros  treináveis
    params_treinaveis =  sum(
        p.numel () for p in modelo.parameters () if p.requires_grad
    )
    params_total = sum (p.numel () for p in modelo.parameters ())

    print(f"LoRA  aplicado! { n_substituidos} módulos substituídos.")
    print(f"Parâmetros  treináveis: { params_treinaveis :,} "
        f"({100* params_treinaveis /params_total :.2 f}%)")

    return modelo


# Demonstração
modelo_base = MiniLMPT(CFG_MINILM)
modelo_lora = aplicar_lora_ao_modelo (
    modelo_base ,
    rank=8,
    alpha =16,
    modulos_alvo =['W_q', 'W_v', 'W_k', 'W_o' ]
)

# Verificar
n_total = sum(p.numel () for p in modelo_lora.parameters ())
n_train = sum(p.numel () for p in modelo_lora.parameters ()
                if p.requires_grad)
print(f"\nVerificação:" )
print(f"  Total: {n_total /1e6:.1f}M parâmetros" )
print(f"  Treináveis: {n_train /1e6:.2f}M ({100* n_train/n_total :.2f}%)")
