import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean  Square Layer Normalization.

    Referência:
        Zhang & Sennrich  (2019). Root Mean  Square Layer Normalization.
        NeurIPS  2019.

    Parâmetros:
        d_model: dimensão a normalizar
        eps: epsilon  para estabilidade numérica
    """

    def __init__(self , d_model: int , eps: float = 1e-6):
        super ().__init__ ()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones( d_model))

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        """
        Parâmetros:
            x: tensor de qualquer shape , normalizado na última  dimensão
        Retorna:
            x normalizado , mesmo shape
        """
        # Calcular  RMS: raiz da média dos  quadrados
        rms = x. pow (2).mean(dim=-1, keepdim=True).add(self.eps).sqrt ()
        return x / rms * self.gamma
