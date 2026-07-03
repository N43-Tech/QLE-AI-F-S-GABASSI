import torch
import torch.nn as nn
import math

def flash_attention_simulado (Q: torch.Tensor ,
                                K: torch.Tensor ,
                                V: torch.Tensor ,
                                tamanho_bloco_seq : int = 64) -> torch.Tensor:
    """
    Simulacao  pedagogica do FlashAttention (tiling + online  softmax).

    Q, K, V: tensores de dimensao (T, d_head)
    tamanho_bloco_seq : tamanho do bloco  processado na "SRAM"
    """
    T, d = Q.shape
    escala = 1.0 / math.sqrt(d)

    # 1) Inicializar  acumuladores de saida e termos do online  softmax
    O = torch.zeros_like(Q)                       # Saida (T, d)
    m = torch.full ((T, 1), float('-inf'), device=Q.device)   # Maximo acumulado (T, 1)
    d_den = torch.zeros ((T, 1), device=Q.device) # Denominador (T, 1)

    # Numero de blocos
    num_blocos_Q = math.ceil(T / tamanho_bloco_seq )
    num_blocos_K = math.ceil(T / tamanho_bloco_seq )

    # 2) Loop  externo sobre blocos de Q
    for i in range (num_blocos_Q):
        inicio_q = i * tamanho_bloco_seq
        fim_q = min (inicio_q + tamanho_bloco_seq , T)
        Q_bloco = Q[inicio_q:fim_q]               # ( B_q , d)

        m_bloco_q = m[inicio_q:fim_q]             # ( B_q , 1)
        d_bloco_q = d_den[inicio_q:fim_q]         # ( B_q , 1)
        O_bloco = O[inicio_q:fim_q]               # ( B_q , d)

        # Loop  interno sobre blocos de K e V
        for j in range (num_blocos_K):
            inicio_k = j * tamanho_bloco_seq
            fim_k = min (inicio_k + tamanho_bloco_seq , T)

            K_bloco = K[inicio_k:fim_k]           # ( B_k , d)
            V_bloco = V[inicio_k:fim_k]           # ( B_k , d)

            # Calcular  logits locais de afinidade
            # (B_q , d) x (d, B_k) -> (B_q , B_k)
            S_bloco = torch.matmul(Q_bloco , K_bloco.t()) * escala

            # Aplicar  mascara causal se j > i
            # (opcional , dependendo de i e j)
            if j * tamanho_bloco_seq  > (i + 1) * tamanho_bloco_seq :
                continue

            # Calcular  maximo local por linha
            m_novo_bloco = torch. max(S_bloco , dim =-1, keepdim=True)[0]   # (B_q , 1)

            # Calculo do expoente  local
            S_exp = torch.exp(S_bloco - m_novo_bloco)                   # ( B_q , B_k)
            d_novo_bloco = torch. sum(S_exp , dim=-1, keepdim=True)       # (B_q , 1)

            # Atualizar  online softmax
            m_proximo = torch. max(m_bloco_q , m_novo_bloco)              # (B_q , 1)

            escala_velho = torch.exp(m_bloco_q - m_proximo)
            escala_novo = torch.exp(m_novo_bloco - m_proximo)

            d_proximo = d_bloco_q * escala_velho + d_novo_bloco * escala_novo

            # Atualizar  acumulador de saida O
            O_bloco = (O_bloco * d_bloco_q * escala_velho +
                        torch.matmul(S_exp * escala_novo , V_bloco)) / d_proximo

            # Atualizar  variaveis de estado
            m_bloco_q = m_proximo
            d_bloco_q = d_proximo

        # Salvar  blocos de volta
        O[inicio_q:fim_q] = O_bloco
        m[inicio_q:fim_q] = m_bloco_q
        d_den[inicio_q:fim_q] = d_bloco_q

    return O
