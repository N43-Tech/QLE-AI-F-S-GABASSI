import torch
import torch.nn as nn
import torch.nn.functional as F

class RoteamentoComCapacidade (nn.Module):
    """
    Roteador MoE com  limites estaticos de capacidade  para especialistas.

    Previne gargalos de memoria e desbalanceamento em hardware  distribuido.
    """
    def __init__(self ,
                d_model: int ,
                n_especialistas : int = 8,
                capacity_factor : float = 1.0):
        super ().__init__ ()
        self.n_especialistas = n_especialistas
        self.capacity_factor = capacity_factor
        self.roteador = nn.Linear(d_model , n_especialistas , bias=False)

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) -> achatar  para (N, D) onde N = B * S
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        N = x_flat.shape [0]

        # 1) Calcular  capacidade do especialista C
        # Usando top -1 para fins de demonstracao
        C = int (self. capacity_factor * (N / self.n_especialistas ))
        C = max (C, 1) # Garantir pelo menos capacidade 1

        # 2) Computar  logits e escolher top -1 especialista
        logits = self.roteador(x_flat)   # (N, E)
        probs = F.softmax(logits , dim=-1)   # (N, E)
        valores_gate , indices_gate = torch. max( probs , dim=-1) # (N,)

        # 3) Alocar  tensores de saida e rastrear ocupacao
        saida = torch.zeros_like(x_flat)
        contagem_especialistas = torch.zeros(self.n_especialistas , dtype=torch.long)

        # Rastrear  quais tokens foram processados
        tokens_processados = 0

        for i in range (N):
            esp_idx = indices_gate[i]. item ()
            peso = valores_gate[i]

            # Verificar se especialista ja atingiu capacidade maxima
            if  contagem_especialistas [esp_idx] < C:
                # Simular computacao pelo especialista esp_idx
                # Em producao: despachar em batch para otimizacao
                contagem_especialistas [esp_idx] += 1
                saida[i] = x_flat[i] * peso   # Representacao simplificada
                tokens_processados += 1
            else :
                # Caso atinja a capacidade: token e DESCARTADO
                # Copia direta da entrada ( simulando a skip  connection)
                saida[i] = x_flat[i]

        taxa_descarte = 1.0 - ( tokens_processados / N)
        # Retorna a saida  remontada para o shape (B, S, D) e estatisticas
        return  saida.view(B, S, D), taxa_descarte
