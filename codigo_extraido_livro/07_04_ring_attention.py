import torch
import torch.distributed as dist

def ring_attention_forward (q_local , k_local , v_local):
    """
    Executa Ring  Attention distribuido sobre P GPUs.
    q_local , k_local , v_local: (seq_por_gpu , n_heads , d_head)
    """
    rank = dist.get_rank ()
    world_size = dist. get_world_size ()

    # Destinos de envio e recebimento no anel
    src_rank = (rank - 1) % world_size
    dst_rank = (rank + 1) % world_size

    # Acumuladores  locais de softmax online
    m_acum = torch.full (( q_local.shape [0], q_local.shape [1], 1), float('-inf'), device=q_local.device)
    d_acum = torch.zeros (( q_local.shape [0], q_local.shape [1], 1), device=q_local.device)
    o_acum = torch.zeros_like(q_local)

    # buffers  para transmissao assincrona
    k_send = k_local.clone ()
    v_send = v_local.clone ()

    k_recv = torch.zeros_like(k_local)
    v_recv = torch.zeros_like(v_local)

    k_atual = k_local
    v_atual = v_local

    for step in range (world_size):
        # Disparar  envio e recebimento assincronos dos proximos blocos KV
        req_k = dist.isend(k_send , dst_rank)
        req_v = dist.isend(v_send , dst_rank)

        req_recv_k = dist.irecv(k_recv , src_rank)
        req_recv_v = dist.irecv(v_recv , src_rank)

        # --- Computacao  paralela do passo  atual ---
        # scores: (seq , heads , seq)
        scores = torch.matmul(q_local , k_atual.transpose (-2,-1)) / (q_local.shape [-1] ** 0.5)
        m_bloco = scores. max(dim=-1, keepdim=True) [0]
        s_exp = torch.exp(scores - m_bloco)
        d_bloco = s_exp. sum(dim=-1, keepdim=True)
        o_bloco = torch.matmul(s_exp , v_atual)

        # Atualizacao do softmax  online
        m_novo = torch. max(m_acum , m_bloco)
        escala_velho = torch.exp(m_acum - m_novo)
        escala_novo = torch.exp(m_bloco - m_novo)

        d_novo = d_acum * escala_velho + d_bloco * escala_novo
        o_acum = (o_acum * d_acum * escala_velho + o_bloco * escala_novo) / (d_novo + 1e-10)

        m_acum = m_novo
        d_acum = d_novo

        # Esperar  finalizacao da comunicacao  para o proximo passo
        req_k.wait ()
        req_v.wait ()
        req_recv_k.wait ()
        req_recv_v.wait ()

        # Preparar  buffers para o proximo  passo
        k_atual = k_recv.clone ()
        v_atual = v_recv.clone ()

        k_send = k_recv.clone ()
        v_send = v_recv.clone ()

    return o_acum
