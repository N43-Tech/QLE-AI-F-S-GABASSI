import torch
import torch.nn as nn
import torch.nn.functional as F

class AtencaoJanelaDeslizante (nn.Module):
    """
    Sliding Window  Attention como no Mistral -7B.

    Combina atenção local (janela  deslizante) com
    FlashAttention  para eficiência de memória.

    Parâmetros:
        d_model: dimensão do modelo
        n_cabecas: cabeças de query
        n_cabecas_kv: cabeças KV (GQA)
        janela: tamanho da janela de atenção W
        comprimento_maximo : comprimento total má ximo
    """

    def __init__(self ,
                d_model: int ,
                n_cabecas: int ,
                n_cabecas_kv: int ,
                janela: int = 4096,
                comprimento_maximo : int = 32768):
        super ().__init__ ()
        self.n_cabecas = n_cabecas
        self.n_cabecas_kv = n_cabecas_kv
        self.n_grupos = n_cabecas // n_cabecas_kv
        self.d_cabeca = d_model // n_cabecas
        self.janela = janela

        self.W_q = nn.Linear(d_model , n_cabecas * self.d_cabeca , bias=False)
        self.W_k = nn.Linear(d_model , n_cabecas_kv * self.d_cabeca , bias=False)
        self.W_v = nn.Linear(d_model , n_cabecas_kv * self.d_cabeca , bias=False)
        self.W_o = nn.Linear(n_cabecas * self.d_cabeca , d_model , bias=False)

        self.rope = RoPE(self.d_cabeca , comprimento_maximo )
        self.escala = self.d_cabeca **   -0.5

    def _mascara_janela (self , T: int ,
                            device: torch.device) -> torch.Tensor:
        """
        Cria máscara  causal restrita à janela deslizante.

        Para posição i, token j é mascarado se:
            j > i (causalidade) OU i - j >= janela (fora da janela)
        """
        mascara = torch.full ((T, T), float ('-inf'), device=device)
        # Máscara  causal: zera triângulo  inferior
        for i in range (T):
            inicio =  max(0, i - self.janela + 1)
            mascara[i, inicio:i+1] = 0.0
        return  mascara.unsqueeze (0).unsqueeze (0) # (1, 1, T, T)

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        Q = self.W_q(x).view(B, T, self.n_cabecas , self.d_cabeca).transpose (1, 2)
        K = self.W_k(x).view(B, T, self.n_cabecas_kv , self.d_cabeca).transpose (1, 2)
        V = self.W_v(x).view(B, T, self.n_cabecas_kv , self.d_cabeca).transpose (1, 2)

        Q = self.rope(Q)
        K = self.rope(K)

        # GQA: expandir KV
        if self.n_grupos > 1:
            K = K. repeat_interleave (self.n_grupos , dim =1)
            V = V. repeat_interleave (self.n_grupos , dim =1)

        mascara = self. _mascara_janela (T, x.device)
        escores = torch.matmul(Q, K.transpose (-2, -1)) * self.escala + mascara
        pesos = F.softmax(escores , dim=-1)

        saida = torch.matmul(pesos , V)
        saida = saida.transpose (1, 2).contiguous ().view(B, T, -1)
        return  self.W_o(saida)
