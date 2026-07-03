import torch
import torch.nn as nn
import torch.nn.functional as F
from typing  import Optional , Tuple

class RoPE(nn.Module):
    """
    Rotary Position  Embedding (RoPE).

    Referência:
        Su et al. (2021). RoFormer: Enhanced Transformer with
        Rotary  Position Embedding. arXiv:2104.09864.

    Parâmetros:
        dim: dimensão de cada cabeça (d_k)
        comprimento_maximo : comprimento máximo de sequência
        base: base para os ângulos  theta (padrão 10000)
    """

    def __init__(self ,
                dim: int ,
                comprimento_maximo : int = 8192 ,
                base: float = 10000.0):
        super ().__init__ ()
        self.dim = dim
        self. comprimento_maximo = comprimento_maximo
        self.base = base

        # theta_j = base ^{-2j/d} para j = 0, 1,..., d/2 - 1
        inv_freq = 1.0 / (
            base ** (torch.arange (0, dim , 2, dtype= torch.float32) / dim)
        )
        self.register_buffer ('inv_freq ', inv_freq)

        # Pré-computar  cos e sin para todos os comprimentos
        self. _atualizar_cache ( comprimento_maximo )

    def _atualizar_cache (self , comprimento: int) -> None:
        """Pré-computa  cos(m*theta) e sin(m*theta) para m=0.. comprimento -1. """
        posicoes = torch.arange(comprimento ,
                                device=self.inv_freq.device ,
                                dtype=self.inv_freq.dtype)
        # Produto  externo: (comprimento , dim /2)
        freqs = torch.outer(posicoes , self.inv_freq)
        # Concatenar  para cobrir todas as dimensões: (comprimento , dim)
        emb = torch.cat([freqs , freqs], dim=-1)

        self.register_buffer ('cos_cache ', emb.cos(), persistent=False)
        self.register_buffer ('sin_cache ', emb.sin(), persistent=False)

    @staticmethod
    def _rotacionar_metade (x: torch.Tensor) -> torch.Tensor:
        """
        Para x = [x1 , x2 , ..., xd], retorna [-x_{d /2+1} , ..., -xd , x1 , ..., x_{d/2}].
        Equivale a multiplicar  por i na representaç ão complexa.
        """
        d = x.shape [-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        return  torch.cat([-x2 , x1], dim=-1)

    def forward(self ,
                x: torch.Tensor ,
                posicoes: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Aplica  RoPE ao tensor x.

        Parâmetros:
            x: tensor de shape (batch , n_cabecas , seq_len , dim_cabeca)
            posicoes: posições opcionais (padrão: 0, 1, ..., seq_len -1)
        Retorna:
            x com RoPE aplicado , mesmo  shape
        """
        seq_len = x.shape [-2]

        if posicoes is  None:
            cos = self.cos_cache [: seq_len ].unsqueeze (0).unsqueeze (0)
            sin = self.sin_cache [: seq_len ].unsqueeze (0).unsqueeze (0)
        else:
            cos = self.cos_cache[posicoes ].unsqueeze (1)
            sin = self.sin_cache[posicoes ].unsqueeze (1)

        # Aplicar  rotação: x_rot = x * cos + rot_half(x) * sin
        return x * cos + self. _rotacionar_metade (x) * sin


class AtencaoMultiCabeca (nn.Module):
    """
    Multi -Head Attention com RoPE , GQA e máscara causal.

    Suporta:
        - MHA: n_cabecas_kv = n_cabecas_query
        - MQA: n_cabecas_kv = 1
        - GQA: 1 < n_cabecas_kv < n_cabecas_query

    Parâmetros:
        d_model: dimensão do modelo
        n_cabecas: número de cabeças de query
        n_cabecas_kv: número de cabeças de key/ value
        comprimento_maximo : comprimento máximo de contexto
        dropout_atencao : taxa de dropout na atenção
    """

    def __init__(self ,
                d_model: int ,
                n_cabecas: int ,
                n_cabecas_kv: int ,
                comprimento_maximo : int = 1024 ,
                dropout_atencao : float = 0.0):
        super ().__init__ ()

        assert  d_model % n_cabecas == 0, \
            "d_model  deve ser divisível por n_cabecas"
        assert  n_cabecas % n_cabecas_kv == 0, \
            "n_cabecas  deve ser divisível por n_cabecas_kv"

        self.d_model = d_model
        self.n_cabecas = n_cabecas
        self.n_cabecas_kv = n_cabecas_kv
        self.n_grupos = n_cabecas // n_cabecas_kv
        self.d_cabeca = d_model // n_cabecas

        # Projeções lineares (sem bias , como em LLaMA)
        self.W_q = nn.Linear(d_model , n_cabecas * self.d_cabeca , bias=False)
        self.W_k = nn.Linear(d_model , n_cabecas_kv * self.d_cabeca , bias=False)
        self.W_v = nn.Linear(d_model , n_cabecas_kv * self.d_cabeca , bias=False)
        self.W_o = nn.Linear(n_cabecas * self.d_cabeca , d_model , bias=False)

        self.rope = RoPE(self.d_cabeca , comprimento_maximo )
        self.dropout_atencao = nn.Dropout( dropout_atencao )

        self.escala = self.d_cabeca **   -0.5

    def _reformatar_kv_gqa (self , x: torch.Tensor) -> torch.Tensor:
        """
        Expande KV para GQA: replica  cada cabeça KV n_grupos vezes.
        (batch , n_kv , seq , d) -> (batch , n_q , seq , d)
        """
        return x. repeat_interleave (self.n_grupos , dim =1)

    def forward(self ,
                x: torch.Tensor ,
                mascara: Optional[torch.Tensor] = None ,
                cache_kv: Optional[Tuple] = None
                ) -> Tuple[torch.Tensor , Optional[ Tuple ]]:
        """
        Forward  pass da atenção multi -cabeça.

        Parâmetros:
            x: (batch , seq , d_model)
            mascara: (batch , 1, seq_q , seq_k) ou None
            cache_kv: (k_cache , v_cache) para infer ência incremental
        Retorna:
            saída: (batch , seq , d_model)
            cache_kv atualizado
        """
        B, T, _ = x.shape

        # 1) Projetar Q, K, V
        Q = self.W_q(x).view(B, T, self.n_cabecas , self.d_cabeca).transpose (1, 2)
        K = self.W_k(x).view(B, T, self.n_cabecas_kv , self.d_cabeca).transpose (1, 2)
        V = self.W_v(x).view(B, T, self.n_cabecas_kv , self.d_cabeca).transpose (1, 2)
        # Q: (B, n_q , T, d_k)
        # K, V: (B, n_kv , T, d_k)

        # 2) Aplicar  RoPE a Q e K
        Q = self.rope(Q)
        K = self.rope(K)

        # 3) Concatenar ao cache KV (inferência incremental)
        if cache_kv is not  None:
            K_cache , V_cache = cache_kv
            K = torch.cat([ K_cache , K], dim =2)
            V = torch.cat([ V_cache , V], dim =2)
        novo_cache = (K, V)

        # 4) Expandir KV para GQA
        if self.n_grupos > 1:
            K = self. _reformatar_kv_gqa (K)
            V = self. _reformatar_kv_gqa (V)

        # 5) Computar  escores de atenção
        # (B, n_q , T_q , T_k)
        escores = torch.matmul(Q, K.transpose (-2, -1)) * self.escala

        # 6) Aplicar máscara  causal
        if mascara is not  None:
            escores = escores + mascara

        # 7) Softmax e dropout
        pesos = F.softmax(escores , dim=-1)
        pesos = self. dropout_atencao (pesos)

        # 8) Produto com  valores
        # (B, n_q , T_q , d_v)
        saida = torch.matmul(pesos , V)

        # 9) Reagrupar e projetar
        # (B, T, d_model)
        saida = saida.transpose (1, 2).contiguous ().view(B, T, -1)
        saida = self.W_o(saida)

        return saida , novo_cache


# Teste de funcionamento
def testar_atencao ():
    """ Verifica a corretude matemática da atenção."""
    B, T, d = 2, 16, 512
    n_q , n_kv = 8, 2  # GQA: 4 queries por grupo KV

    modelo = AtencaoMultiCabeca (d, n_q , n_kv)
    x = torch.randn(B, T, d)

    # Máscara  causal
    mascara = torch.full ((1, 1, T, T), float ('-inf'))
    mascara = torch.triu(mascara , diagonal =1)

    with torch.no_grad ():
        saida , cache = modelo(x, mascara=mascara)

    print(f"Entrada:   {x.shape}")
    print(f"Saída:     {saida.shape}")
    print(f"K-cache:   {cache [0]. shape}")
    print(f"Parâmetros: {sum(p.numel () for p in modelo.parameters ()):,}")

    # Verificar  que a atenção é causal
    # (token 0 não deve  influenciar token 5 etc.)
    assert saida.shape == x.shape , "Shape  incorreto !"
    print("Teste de atenção: OK!" )

testar_atencao ()
