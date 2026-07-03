import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceiverResampler (nn.Module):
    """
    Comprime N_patches de dimensao  d_vit em M tokens de dimensao  d_model
    usando atencao  cruzada com consultas  aprendidas.
    """
    def __init__(self ,
                d_vit: int = 1024,
                d_model: int = 512,
                n_tokens_saida : int = 64,
                n_heads: int = 8):
        super ().__init__ ()
        self.n_tokens_saida = n_tokens_saida
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # 1) Consultas  latentes aprendidas (Q)
        self. queries_latentes = nn.Parameter(torch.randn(n_tokens_saida , d_model) * 0.02)

        # 2) Projeções lineares
        self.proj_q = nn.Linear(d_model , d_model , bias=False)
        self.proj_k = nn.Linear(d_vit , d_model , bias=False)
        self.proj_v = nn.Linear(d_vit , d_model , bias=False)

        # Projeção de saída
        self.proj_out = nn.Linear(d_model , d_model)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_vit)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self , x_kv: torch.Tensor) -> torch.Tensor:
        # x_kv: (B, N_patches , d_vit)
        B, N, _ = x_kv.shape

        # Aplicar  LayerNorm nas entradas
        x_kv = self.ln_kv(x_kv)

        # Expandir  consultas latentes para o tamanho do batch: (B, M, d_model)
        q_lat = self. queries_latentes .unsqueeze (0).expand(B, -1, -1)
        q_lat = self.ln_q(q_lat)

        # Projetar Q, K, V
        # Q: (B, M, H, d_head)
        Q = self.proj_q(q_lat).view(B, self.n_tokens_saida , self.n_heads , self.d_head).transpose (1, 2)
        # K, V: (B, N, H, d_head)
        K = self.proj_k(x_kv).view(B, N, self.n_heads , self.d_head).transpose (1, 2)
        V = self.proj_v(x_kv).view(B, N, self.n_heads , self.d_head).transpose (1, 2)

        # 3) Scaled Dot -Product Cross -Attention
        # similaridades: (B, H, M, N)
        scores = torch.matmul(Q, K.transpose (-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(scores , dim=-1)

        # saida de atencao: (B, H, M, d_head) -> (B, M, d_model)
        contexto = torch.matmul(attn , V).transpose (1, 2).contiguous ()
        contexto = contexto.view(B, self.n_tokens_saida ,-1)

        # Projeção final e conexão residual   com as consultas
        saida = self.ln_out(self.proj_out(contexto) + q_lat)
        return  saida


class ProjetorVisual (nn.Module):
    """
    Projetor MLP que mapeia  features visuais  para o espaço do LLM.

    Equivalente ao MLP  connector do LLaVA -1.5.
    """
