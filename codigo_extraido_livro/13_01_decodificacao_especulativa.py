import torch
import torch.nn.functional as F
from typing  import Tuple

@torch.no_grad ()
def decodificacao_especulativa (
    modelo_alvo ,
    modelo_rascunho ,
    ids_prompt: torch.Tensor ,
    max_novos: int = 200,
    gamma: int = 4,
    temperatura: float  = 1.0,
    dispositivo: str  = 'cpu'
) -> Tuple[torch.Tensor , dict ]:
    """
    Decodificação especulativa (Leviathan et al., 2023).

    Usa modelo_rascunho  para propor gamma  tokens e
    modelo_alvo  para verificar e corrigir.

    Parâmetros:
        modelo_alvo: LLM grande (pi_target)
        modelo_rascunho : LLM pequeno e rápido ( pi_draft)
        ids_prompt: (1, T) -- tokens do prompt
        max_novos: máximo de tokens a gerar
        gamma: tokens  propostos por rodada
        temperatura: temperatura de amostragem
    Retorna:
        ids_gerados: tensor com tokens   gerados
        stats: estatísticas de aceitação
    """
    ids = ids_prompt.to(dispositivo)
    cache_rascunho = None
    n_aceitos_total = 0
    n_rodadas = 0

    while ids.shape [1] - ids_prompt.shape [1] < max_novos:
        T_atual = ids.shape [1]

        # =============================================
        # PASSO 1: Draft  model propõe gamma  tokens
        # =============================================
        tokens_rascunho = []
        probs_rascunho = []
        ids_temp = ids.clone ()
        cache_temp = cache_rascunho

        for _ in range (gamma):
            if cache_temp  is None:
                entrada = ids_temp
            else :
                entrada = ids_temp [:, -1:]

            logits_r , cache_temp = modelo_rascunho (
                entrada , cache_kvs=cache_temp , retornar_cache =True
            )

            probs_r = F.softmax(logits_r [:, -1, :] / temperatura , dim=-1)
            token_r = torch.multinomial(probs_r , num_samples =1)

            tokens_rascunho .append(token_r)
            probs_rascunho .append(probs_r)
            ids_temp = torch.cat([ ids_temp , token_r], dim =1)

        # Sequência  proposta pelo rascunho
        ids_proposta = ids_temp   # (1, T_atual + gamma)

        # =============================================
        # PASSO 2: Target  model verifica  TODOS de uma vez
        # =============================================
        logits_alvo , _ = modelo_alvo(ids_proposta)
        # Logits  para as posições dos tokens rascunho
        # (posições T_atual a T_atual+gamma -1 + o próximo)
        logits_verif = logits_alvo [:, T_atual -1: T_atual+gamma , :]

        probs_alvo = F.softmax(logits_verif / temperatura , dim=-1)

        # =============================================
        # PASSO 3: Aceitar/rejeitar especulativamente
        # =============================================
        n_aceitos = 0

        for k in range (gamma):
            token_k = tokens_rascunho [k]
            prob_alvo_k  = probs_alvo [:, k, :].gather(-1, token_k)
            prob_rascunho_k = probs_rascunho[k].gather(-1, token_k)

            # Taxa de aceitação
            taxa = torch. min(
                torch.ones_like(prob_alvo_k),
                prob_alvo_k / ( prob_rascunho_k + 1e-10)
            )

            u = torch.rand_like(taxa)

            if u <= taxa:
                # Aceito!
                ids = torch.cat([ids , token_k], dim =1)
                n_aceitos += 1
            else :
                # Rejeitado! Amostrar do target corrigido
                prob_corrigida = F.relu(
                    probs_alvo [:, k, :] - probs_rascunho [k]
                )
                prob_corrigida = prob_corrigida / prob_corrigida .sum()
                token_corrigido = torch.multinomial (prob_corrigida , 1)
                ids = torch.cat([ids , token_corrigido ], dim =1)
                break  # Parar na primeira  rejeição

        if n_aceitos == gamma:
            # Todos  aceitos: adicionar  token bonus do target
            token_bonus = torch.multinomial( probs_alvo [:, gamma , :], 1)
            ids = torch.cat([ids , token_bonus], dim =1)
            cache_rascunho = cache_temp

        n_aceitos_total += n_aceitos
        n_rodadas += 1

        # Verificar  EOS
        if ids[0,  -1]. item () == 3:
            break

    stats = {
        'tokens_gerados ': ids.shape [1] - ids_prompt.shape [1],
        'rodadas ': n_rodadas ,
        'taxa_aceitacao ': n_aceitos_total / ( n_rodadas * gamma),
        'tokens_por_rodada ': (ids.shape [1] - ids_prompt.shape [1]) / max(n_rodadas , 1)
    }

    return ids , stats
