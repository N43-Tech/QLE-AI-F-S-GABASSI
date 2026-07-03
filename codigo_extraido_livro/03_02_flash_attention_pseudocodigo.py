def flash_attention_forward (Q, K, V):
    # Q: (T, d), K: (T, d), V: (T, d) na HBM
    # d = dimensao da cabeca , T = seq_len
    # Br , Bc: tamanhos dos blocos na SRAM (ex: Br =64, Bc =64)
    Tr = T // Br
    Tc = T // Bc

    O = zeros_like(Q)  # Matriz de saida na HBM
    L = zeros(T)       # Acumulador de somas exp na HBM
    M = fill(-inf , T) # Acumulador de maximos na HBM

    # Loop externo: processa  blocos de Queries
    for i in range (Tr):
        Q_i = carregar_bloco_hbm_para_sram (Q[i*Br : (i+1)*Br]) # (Br , d)

        # Loop  interno: processa blocos de Keys/ Values
        for j in range (Tc):
            K_j = carregar_bloco_hbm_para_sram (K[j* Bc : (j+1)*Bc]) # (Bc , d)
            V_j = carregar_bloco_hbm_para_sram (V[j* Bc : (j+1)*Bc]) # (Bc , d)

            # 1) Calcular  scores locais na SRAM: ( Br , Bc)
            S_ij = Q_i @ K_j.T / sqrt(d)

            # 2) Softmax  online incremental  para o bloco
            m_bloco =  max(S_ij , axis =-1)  # (Br ,)
            l_bloco =  sum(exp(S_ij - m_bloco), axis =-1) # (Br ,)

            m_novo =  max(M[i*Br : (i+1)*Br], m_bloco)
            l_novo = exp(M[i*Br : (i+1)*Br] - m_novo) * L[i*Br : (i+1)*Br] + exp( m_bloco - m_novo) * l_bloco

            # 3) Atualizar  saida parcial  acumulada na SRAM
            P_ij = exp(S_ij - m_bloco)
            O_bloco = P_ij @ V_j  # (Br , d)

            O_acum = O[i*Br : (i+1)*Br]
            O_novo = (O_acum * (L[i*Br : (i+1)*Br] * exp(M[i*Br : (i+1)*Br] - m_novo)).unsqueeze (-1) +
                        O_bloco * exp(m_bloco - m_novo).unsqueeze (-1)) / l_novo.unsqueeze (-1)

            # Gravar  valores intermediarios na HBM
            M[i*Br : (i+1)*Br] = m_novo
            L[i*Br : (i+1)*Br] = l_novo
            O[i*Br : (i+1)*Br] = O_novo

    return O
