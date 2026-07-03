import torch
import torch.nn as nn
import numpy as np
from typing import List , Dict , Tuple

class EstimadorBradleyTerry (nn.Module):
    """
    Estima os scores  latentes dos modelos  usando Bradley -Terry MLE.
    Equivalente a encontrar a pontuacao  Elo ideal a partir de batalhas do Arena.
    """
    def __init__(self , n_modelos: int):
        super ().__init__ ()
        # Inicializar  scores latentes com zeros ( habilidade igual)
        self.scores = nn.Parameter(torch.zeros( n_modelos))

    def forward(self , idx_vencedor: torch.Tensor , idx_perdedor: torch.Tensor) -> torch.Tensor:
        # P(vence) = sigmoid( score_vencedor - score_perdedor )
        logits = self.scores[idx_vencedor] - self.scores[idx_perdedor]
        # Queremos  maximizar log sigmoid(logits), que e o mesmo que minimizar  binary cross entropy
        # com targets = 1
        perda = -torch.log(torch.sigmoid(logits) + 1e-12).mean ()
        return  perda

def estimar_elo_batalhas (batalhas: List[Tuple[ int , int]], n_modelos:  int , max_iter: int = 2000) -> np.ndarray:
    """
    Recebe batalhas  como lista de pares ( vencedor_idx , perdedor_idx) e calcula  Elo ratings.
    """
    vencedores = torch.tensor ([b[0] for b in batalhas], dtype=torch. long)
    perdedores = torch.tensor ([b[1] for b in batalhas], dtype=torch. long)

    modelo = EstimadorBradleyTerry (n_modelos)
    otimizador = torch.optim.LBFGS(modelo.parameters (), max_iter=max_iter , lr =0.1)

    # Adicionar  restricao: o primeiro modelo (ou a media) serve  como ancora (score = 0)
    def closure ():
        otimizador.zero_grad ()
        perda = modelo(vencedores , perdedores)
        # Forçar média dos scores a ser zero ( ancoragem)
        perda += 0.01 * torch.mean(modelo.scores) ** 2
        perda.backward ()
        return  perda

    otimizador.step(closure)

    # Converter  scores latentes theta para  ratings Elo  tradicionais
    with torch.no_grad ():
        scores_theta = modelo.scores.numpy ()

    # Ancorar o rating Elo  baseline em 1000
    elo_ratings = 1000.0 + (400.0 / np.log (10.0)) * (scores_theta - np.mean(scores_theta))
    return elo_ratings


def bootstrap_intervalo_confianca (dados_acertos: np.ndarray ,
                                    n_reamostragens : int = 10000 ,
                                    confianca: float = 0.95) -> Tuple[float , Tuple[float , float ]]:
    """
    Calcula o intervalo de confianca de um benchmark  via bootstrap nao -parametrico.
    """
    n = len(dados_acertos)
    acuracia_original = np.mean(dados_acertos)

    # Reamostragem  com reposicao
    reamostras = np.random.choice(dados_acertos , size =( n_reamostragens , n), replace=True)
    acuracias_boot = np.mean(reamostras , axis =1)

    # Calcular  percentis
    alpha_inf = (1.0 - confianca) / 2.0
    alpha_sup = 1.0 - alpha_inf

    ic_inf = np.percentile(acuracias_boot , alpha_inf * 100)
    ic_sup = np.percentile(acuracias_boot , alpha_sup * 100)

    return acuracia_original , (ic_inf , ic_sup)
