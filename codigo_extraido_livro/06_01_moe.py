import torch
import torch.nn as nn
import torch.nn.functional as F
from typing  import Tuple

class CamadaMoE(nn.Module):
    """
    Camada Mixture of Experts  com roteamento Top -K.

    Substitui a camada FFN no bloco  Transformer.
    Cada especialista é uma FFN SwiGLU   independente.

    Parâmetros:
        d_model: dimensão do modelo
        n_especialistas : número total de especialistas (E)
        top_k: especialistas  ativados por token (k)
        d_ff_especialista : dimensão interna de cada especialista
        coef_balanceamento : peso da loss de balanceamento (alpha)
        capacidade_fator : capacidade = ceil(top_k/E * T * fator)
    """

    def __init__(self ,
                d_model: int ,
                n_especialistas : int = 8,
                top_k: int = 2,
                d_ff_especialista : int = 2048,
                coef_balanceamento : float = 0.01 ,
                capacidade_fator : float = 1.25):
        super ().__init__ ()

        assert  top_k <= n_especialistas , "top_k <= n_especialistas "

        self.d_model = d_model
        self.n_especialistas = n_especialistas
        self.top_k = top_k
        self. coef_balanceamento = coef_balanceamento
        self. capacidade_fator = capacidade_fator

        # Rede de roteamento: projeção simples d_model -> E
        self.roteador = nn.Linear(d_model , n_especialistas , bias=False)

        # E especialistas (FFN SwiGLU)
        d_ff_int = int (2 * d_ff_especialista / 3)
        d_ff_int = 64 * (( d_ff_int + 63) // 64)   # Múltiplo de 64

        self.W_gate = nn.Parameter(
            torch.randn(n_especialistas , d_model , d_ff_int) * 0.02
        )
        self.W_up = nn.Parameter(
            torch.randn(n_especialistas , d_model , d_ff_int) * 0.02
        )
        self.W_down = nn.Parameter(
            torch.randn(n_especialistas , d_ff_int , d_model) * 0.02 / (2*8) **0.5
        )

    def _computar_especialista (self ,
                                x: torch.Tensor ,
                                idx_esp: int ) -> torch.Tensor:
        """ Passa x pelo especialista idx_esp."""
        gate = F.silu(x @ self.W_gate[idx_esp ])
        up   = x @ self.W_up[idx_esp]
        return (gate * up) @ self.W_down[idx_esp]

    def _loss_balanceamento (self ,
                                logits_roteador : torch.Tensor ,
                                indices_top_k: torch.Tensor) -> torch.Tensor:
        """
        Calcula a auxiliary  loss de balanceamento.

        Parâmetros:
            logits_roteador : (T, E) -- logits brutos do roteador
            indices_top_k: (T, k) -- índices dos especialistas selecionados
        """
        T = logits_roteador .shape [0]
        E = self. n_especialistas

        # f_i: fração de tokens por  especialista (n ão diferenciável)
        contagem = torch.zeros(E, device= logits_roteador .device)
        contagem.scatter_add_ (0,
                                indices_top_k.flatten (),
                                torch.ones(T * self.top_k , device= logits_roteador .device))
        f_i = contagem / T

        # P_i: probabilidade média do softmax ( diferenciável)
        probs = F.softmax(logits_roteador , dim=-1) # (T, E)
        P_i = probs.mean(dim =0)  # (E,)

        # Loss: alpha * E * sum(f_i * P_i)
        # Ideal: f_i = P_i = 1/E para  todos ( distribuição uniforme)
        return  self. coef_balanceamento * E * (f_i * P_i).sum()

    def forward(self ,
                x: torch.Tensor
                ) -> Tuple[torch.Tensor , torch.Tensor ]:
        """
        Parâmetros:
            x: (batch , seq , d_model)
        Retorna:
            saída: (batch , seq , d_model)
            loss_bal: escalar (loss de balanceamento)
        """
        B, T_seq , D = x.shape

        # Achatar  para (B*T, D)
        x_flat = x.view(-1, D)
        N = x_flat.shape [0]   # N = B * T_seq

        # 1) Computar  logits de roteamento
        logits = self.roteador(x_flat)     # (N, E)

        # 2) Top -K seleção (não diferenciável)
        top_k_logits , top_k_indices = torch.topk(
            logits , self.top_k , dim=-1
        )  # (N, k)

        # 3) Softmax  sobre os top -k logits ( diferenciável)
        top_k_weights = F.softmax(top_k_logits , dim =-1)  # (N, k)

        # 4) Computar saídas dos  especialistas
        saida = torch.zeros(N, D, device=x.device , dtype=x.dtype)

        for k in range (self.top_k):
            indices_k = top_k_indices [:, k]    # (N,) - especialista para cada  token
            pesos_k   = top_k_weights [:, k]    # (N,) - peso do especialista

            for e in range (self. n_especialistas ):
                # Selecionar tokens roteados  para especialista e
                mascara = (indices_k == e)
                if not mascara.any():
                    continue

                tokens_e = x_flat[mascara]      # ( n_e , D)
                saida_e = self._computar_especialista (tokens_e, e)
                saida[mascara] += pesos_k[mascara ].unsqueeze (-1) * saida_e

        # 5) Loss de balanceamento
        loss_bal = self. _loss_balanceamento (logits , top_k_indices)

        return  saida.view(B, T_seq , D), loss_bal


# Comparação: FFN densa vs MoE
def comparar_params ():
    d = 512
    d_ff = 2048

    # FFN densa (SwiGLU)
    ffn_densa = FFNSwiGLU(d, d_ff)
    n_densa = sum (p.numel () for p in ffn_densa.parameters ())

    # MoE com 8 especialistas , top -2
    moe = CamadaMoE(d, n_especialistas =8, top_k =2, d_ff_especialista =d_ff)
    n_moe_total = sum (p.numel () for p in moe.parameters ())

    print(f"FFN densa:          {n_densa :,} parâ metros")
    print(f"MoE (8 esp , top2): {n_moe_total :,} parâ metros (total)")
    print(f"Parâmetros  ativos: ~{ n_densa * 2:,} ( apenas 2/8 esp)")
    print(f"Fator de expansão: {n_moe_total/n_densa:.1f}x total , "
        f"{2/8:.0%}  ativo por token")

comparar_params ()
