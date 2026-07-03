import torch
import torch.nn as nn
import torch.nn.functional as F

class SSMSeletivo(nn.Module):
    """
    State Space  Model Seletivo (núcleo do Mamba).

    Referência:
        Gu & Dao (2023). Mamba: Linear -Time Sequence Modeling
        with Selective  State Spaces. arXiv:2312.00752.

    Parâmetros:
        d_model: dimensão de entrada
        d_estado: dimensão do estado  interno N
        d_dt: dimensão de dt (projeta  para rank baixo)
        A_rank: rank da matriz A (diagonal se 1)
    """

    def __init__(self ,
                d_model: int ,
                d_estado: int = 16,
                d_dt: int = 64,
                dt_min: float = 0.001 ,
                dt_max: float = 0.1):
        super ().__init__ ()
        self.d_model = d_model
        self.d_estado = d_estado

        # Matriz A: aprendida  como log para garantir Re(A) < 0
        # A diagonal: (d_model , d_estado)
        A = torch.arange (1, d_estado + 1, dtype= torch.float32)
        A = A.unsqueeze (0).expand(d_model , -1)
        self.A_log = nn.Parameter(torch.log(A))

        # B, C: projetadas a partir da entrada ( seleção)
        self.proj_B = nn.Linear(d_model , d_estado , bias=False)
        self.proj_C = nn.Linear(d_model , d_estado , bias=False)

        # dt: passo de discretização (seleção)
        self.proj_dt = nn.Linear(d_model , d_dt , bias=True)
        self.dt_proj = nn.Linear(d_dt , d_model , bias=True)

        # Inicializar  dt_proj bias para [dt_min , dt_max]
        nn.init.uniform_(
            self.proj_dt.bias ,
            math.log(dt_min), math.log(dt_max)
        )

        # Parâmetro D (skip  connection)
        self.D = nn.Parameter(torch.ones(d_model))

    def _varredura_sequencial (self ,
                                u: torch.Tensor ,
                                dt: torch.Tensor ,
                                A: torch.Tensor ,
                                B: torch.Tensor ,
                                C: torch.Tensor) -> torch.Tensor:
        """
        Varredura  SSM sequencial (para inferência ou fallback).

        Parâmetros:
            u: (batch , seq , d_model) - entrada
            dt: (batch , seq , d_model) - passo de tempo
            A: (d_model , d_estado) - matriz de transição
            B: (batch , seq , d_estado) - matriz de entrada
            C: (batch , seq , d_estado) - matriz de saída
        """
        B_batch , T, D = u.shape
        N = self.d_estado

        # Discretização: A_barra = exp(dt * A), B_barra = dt * B
        # dt: (batch , T, D), A: (D, N) -> dA: ( batch , T, D, N)
        dA = torch.exp(dt.unsqueeze (-1) * (-A. abs()))  # estável
        dB = dt.unsqueeze (-1) * B.unsqueeze (2)  # ( batch , T, D, N)

        # Varredura  recorrente
        saidas = torch.zeros(B_batch , T, D, device= u.device)
        h = torch.zeros(B_batch , D, N, device=u.device , dtype=u.dtype)

        for t in range (T):
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze (-1)
            # y_t = C_t @ h_t para cada  dimensão
            y = (h * C[:, t]. unsqueeze (2)).sum(-1)
            saidas [:, t] = y

        return  saidas

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        """
        Parâmetros:
            x: (batch , seq , d_model)
        Retorna:
            y: (batch , seq , d_model)
        """
        B_batch , T, D = x.shape

        # Parâmetros  seletivos (dependem da entrada)
        dt_raw = F.softplus(self.dt_proj(
            F.silu(self.proj_dt(x))
        ))  # (batch , T, D) -- passo de tempo

        B_t = self.proj_B(x)    # (batch , T, N)
        C_t = self.proj_C(x)    # (batch , T, N)

        # Matriz A (fixa , mas aprendida)
        A = -self.A_log. float ().exp() # (D, N), Re (A) < 0

        # Varredura  SSM
        y = self. _varredura_sequencial (x, dt_raw , A, B_t , C_t)

        # Skip  connection com D
        y = y + self.D.unsqueeze (0).unsqueeze (0) * x

        return y


import math

class BlocoMamba(nn.Module):
    """
    Bloco Mamba  completo com projeções de entrada/ saída.
    """

    def __init__(self , d_model: int , d_estado: int = 16,
                fator_expansao : int = 2):
        super ().__init__ ()
        d_inner = d_model * fator_expansao

        self.norma = RMSNorm(d_model)
        self.proj_entrada = nn.Linear(d_model , d_inner * 2, bias=False)
        self.ssm = SSMSeletivo(d_inner , d_estado)
        self.proj_saida = nn.Linear(d_inner , d_model , bias=False)

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norma(x)

        # Projetar e dividir em x e gate
        xz = self.proj_entrada(x)
        x_proj , z = xz.chunk(2, dim=-1)

        # SSM na ramificação principal
        x_ssm = self.ssm(F.silu(x_proj))

        # Gating com  ramificação z
        saida = x_ssm * F.silu(z)

        return  residual + self.proj_saida(saida)
