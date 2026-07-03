import torch
import torch.nn.functional as F

@torch.no_grad ()
def gerar_texto(modelo ,
                ids_prompt: torch.Tensor ,
                max_novos_tokens : int = 200,
                temperatura: float = 0.8,
                top_p: float = 0.9,
                dispositivo: str = 'cpu') -> str :
    """
    Geração autoregressiva  com nucleus sampling ( top -p).

    Parâmetros:
        modelo: MiniLMPT  treinado
        ids_prompt: (1, T_prompt) -- IDs do prompt
        max_novos_tokens : número máximo de novos tokens
        temperatura: controla  aleatoriedade (>1 = mais diverso)
        top_p: massa de probabilidade  acumulada para nucleus
        dispositivo: dispositivo de inferência
    Retorna:
        lista de IDs gerados
    """
    modelo.eval ()
    ids = ids_prompt.to(dispositivo)
    cache_kvs = None

    for _ in range ( max_novos_tokens ):
        # Primeiro  passo: processa todo o prompt
        # Passos  seguintes: processa apenas o ú ltimo token
        if cache_kvs is  None:
            entrada = ids
        else:
            entrada = ids[:,  -1:]   # Apenas último token

        logits , cache_kvs = modelo(
            entrada ,
            cache_kvs=cache_kvs ,
            retornar_cache =True
        )

        # Logits do último  token
        logits_ultimo = logits [:, -1, :]. float ()

        # Aplicar  temperatura
        logits_escalonados = logits_ultimo /  max ( temperatura , 1e-5)

        # Top -p (nucleus) sampling
        probs = F.softmax(logits_escalonados , dim =-1)

        # Ordenar  probabilidades em ordem decrescente
        probs_ord , indices_ord = torch.sort(probs , dim=-1, descending=True)
        probs_cum = torch.cumsum(probs_ord , dim=-1)

        # Remover  tokens além do nucleus top -p
        remover = probs_cum - probs_ord > top_p
        probs_ord[remover] = 0.0
        probs_ord = probs_ord / probs_ord.  sum(dim =-1, keepdim=True)

        # Amostrar
        idx_amostrado = torch.multinomial(probs_ord, num_samples =1)
        token_novo = torch.gather(indices_ord , -1, idx_amostrado)

        ids = torch.cat([ids , token_novo], dim=-1)

        # Parar no token EOS
        if token_novo.item () == 3:  # ID do  <|eos|>
            break

    return ids [0]. tolist ()
