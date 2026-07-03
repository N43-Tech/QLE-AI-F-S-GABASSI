import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNSwiGLU(nn.Module):
    """
    Feed -Forward Network com ativação SwiGLU.
    Usada em LLaMA , Gemma , Mistral e nosso MiniLM - PT.

    Parâmetros:
        d_model: dimensão de entrada/saída
        d_ff: dimensão interna (antes de 2/3 scaling)
        dropout: taxa de dropout
    """

    def __init__(self ,
                d_model: int ,
                d_ff: int ,
                dropout: float = 0.0):
        super ().__init__ ()
        # Reduzir  d_ff para 2/3 para compensar 3 matrizes vs 2
        d_ff_interno =  int(2 * d_ff / 3)
        # Arredondar  para múltiplo de 256 (eficiê ncia em GPU)
        d_ff_interno = 256 * (( d_ff_interno + 255) // 256)

        self.W_gate = nn.Linear(d_model , d_ff_interno , bias=False)
        self.W_up    = nn.Linear(d_model , d_ff_interno , bias=False)
        self.W_down = nn.Linear(d_ff_interno , d_model , bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        """
        Parâmetros:
            x: (batch , seq , d_model)
        Retorna:
            (batch , seq , d_model)
        """
        # Gate: Swish(W_gate * x)
        gate = F.silu(self.W_gate(x))    # silu = Swish
        # Up: W_up * x
        up   = self.W_up(x)
        # Produto  elemento -a-elemento
        ativacao = gate * up
        ativacao = self.dropout(ativacao)
        # Projetar de volta  para d_model
        return  self.W_down(ativacao)
