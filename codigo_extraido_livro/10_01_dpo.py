import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import  Tuple

def log_prob_sequencia (modelo: nn.Module ,
                        ids_entrada: torch.Tensor ,
                        ids_alvo: torch.Tensor) -> torch.Tensor:
    """
    Calcula log P(y | x) para uma sequência completa.

    Parâmetros:
        modelo: LLM (MiniLMPT)
        ids_entrada: (B, T) -- tokens de entrada ( prompt + resposta)
        ids_alvo: (B, T) -- tokens  alvo (apenas resposta , resto =-100)
    Retorna:
        log_probs: (B,) -- log -probabilidade de cada sequência
    """
    with torch.no_grad () if not modelo.training else  torch.enable_grad ():
        logits , _ = modelo(ids_entrada)

    # Log -softmax sobre vocabulário
    log_probs_por_token = F.log_softmax(logits , dim =-1)   # (B, T, V)

    # Selecionar log -prob dos tokens alvo
    # ids_alvo =  -100 para tokens a ignorar (prompt)
    mascara_resp = (ids_alvo !=  -100)
    ids_alvo_limpos = ids_alvo.clamp( min =0)  # Remover  -100

    log_probs_alvo = log_probs_por_token .gather(
        dim=-1, index= ids_alvo_limpos .unsqueeze (-1)
    ).squeeze (-1)  # (B, T)

    # Somar sobre  tokens da resposta
    log_probs_alvo = log_probs_alvo * mascara_resp.float ()
    return log_probs_alvo .sum(dim=-1)  # (B,)


class TreinadorDPO:
    """
    Treinador  DPO para alinhamento de LLMs.

    Referência:
        Rafailov et al. (2023). Direct  Preference Optimization:
        Your  Language Model is Secretly a Reward Model. NeurIPS 2023.

    Parâmetros:
        politica: modelo a alinhar (pi_theta)
        referencia: modelo de referência   congelado (pi_ref)
        beta: coeficiente de regularização KL
        lr: taxa de aprendizado
    """

    def __init__(self ,
                politica: nn.Module ,
                referencia: nn.Module ,
                beta: float = 0.1,
                lr: float = 5e-7):
        self.politica = politica
        self.referencia = referencia
        self.beta = beta

        # Congelar  modelo de referência
        for p in self.referencia.parameters ():
            p. requires_grad_(False)

        self.otimizador = torch.optim.AdamW(
            self.politica.parameters (), lr=lr , weight_decay =0.0
        )

    def _calcular_logratios (self ,
                                prompt_w: torch.Tensor,
                                resp_w: torch.Tensor ,
                                prompt_l: torch.Tensor,
                                resp_l: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor ]:
        """
        Calcula  log(pi_theta/pi_ref) para  resposta preferida (w) e rejeitada (l).
        """
        # Log -probs da política treinada
        lp_theta_w = log_prob_sequencia (self.politica , prompt_w , resp_w)
        lp_theta_l = log_prob_sequencia (self.politica , prompt_l , resp_l)

        # Log -probs da referência (sem gradiente)
        with  torch.no_grad ():
            lp_ref_w = log_prob_sequencia (self.referencia , prompt_w , resp_w)
            lp_ref_l = log_prob_sequencia (self.referencia , prompt_l , resp_l)

        # Log -ratios: log(pi_theta / pi_ref)
        logratio_w = lp_theta_w - lp_ref_w
        logratio_l = lp_theta_l - lp_ref_l

        return  logratio_w , logratio_l

    def calcular_perda_dpo (self ,
                            prompt_w , resp_w ,
                            prompt_l , resp_l
                            ) -> Tuple[torch.Tensor , dict ]:
        """
        Calcula a DPO loss e métricas  auxiliares.

        Parâmetros:
            prompt_w , resp_w: prompt e resposta preferida (B, T)
            prompt_l , resp_l: prompt e resposta rejeitada (B, T)
        Retorna:
            loss: escalar  DPO
            metricas: dict com métricas de diagnó stico
        """
        logratio_w , logratio_l = self._calcular_logratios (
            prompt_w , resp_w , prompt_l , resp_l
        )

        # DPO loss: -E[log sigma(beta * (logratio_w - logratio_l))]
        margem = self.beta * (logratio_w - logratio_l)
        perda = -F.logsigmoid(margem).mean ()

        # Métricas  diagnósticas
        with  torch.no_grad ():
            metricas = {
                'perda_dpo ':      perda.item (),
                'acuracia ':       (margem > 0).float ().mean ().item (),
                'margem_media ':   margem.mean ().item (),
                'reward_w ':       (self.beta * logratio_w).mean ().item (),
                'reward_l ':       (self.beta * logratio_l).mean ().item (),
            }

        return perda , metricas

    def passo_treinamento (self , lote: dict) -> dict:
        """ Executa um passo de treinamento  DPO."""
        self.politica.train ()
        self.otimizador.zero_grad ()

        perda , metricas = self. calcular_perda_dpo (
            lote[ 'prompt_w '], lote['resp_w '],
            lote[ 'prompt_l '], lote['resp_l ']
        )

        perda.backward ()
        nn.utils.clip_grad_norm_ (self.politica.parameters (), 1.0)
        self.otimizador.step ()

        return  metricas


print("Módulo de treinamento  DPO implementado!" )
print("Métricas a monitorar:" )
print("  - acuracia: deve ser > 50% (e crescer)"  )
print("  - margem_media: deve ser > 0 (e crescer)"  )
print("  - reward_w > reward_l: sempre" )
