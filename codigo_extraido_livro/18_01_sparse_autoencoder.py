import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder (nn.Module):
    """
    Sparse Autoencoder (SAE) para interpretabilidade de LLMs.

    Decompõe ativações de tamanho d em F features esparsas (F >> d).
    """

    def __init__(self , d_model: int , n_features: int , l1_coef: float = 1e-3):
        super ().__init__ ()
        self.d_model = d_model
        self.n_features = n_features
        self.l1_coef = l1_coef

        # Bias do decoder (inicializado  com a média das ativações do corpus)
        self.b_dec = nn.Parameter(torch.zeros( d_model))

        # Encoder  linear: x_cent -> z
        self.W_enc = nn.Parameter(torch.empty( n_features , d_model))
        self.b_enc = nn.Parameter(torch.zeros( n_features))

        # Decoder  linear: z -> x_hat
        self.W_dec = nn.Parameter(torch.empty( d_model , n_features))

        self. reset_parameters ()

        # Rastreamento de passos sem ativação (para ressuscitar neurônios mortos)
        self.register_buffer (" passos_sem_ativacao ", torch.zeros(n_features , dtype=torch.long))

    def reset_parameters (self):
        # Kaiming nos pesos do encoder
        nn.init. kaiming_uniform_ (self.W_enc , a=1.0)
        # Decoder  inicializado com colunas normalizadas (norma L2 = 1)
        nn.init.normal_(self.W_dec , std =1.0)
        with torch.no_grad ():
            self.W_dec.data = F.normalize(self.W_dec.data , dim =0)

    def forward(self , x: torch.Tensor):
        """
        Parâmetros:
            x: tensor de ativações (batch , d_model)
        Retorna:
            x_hat: reconstrução de x
            z: ativações latentes  esparsas
            loss: perda  composta (reconstrução + L1)
        """
        # Centralizar  entrada
        x_cent = x - self.b_dec

        # 1) Codificação com ativação ReLU
        z = F.relu(torch.matmul(x_cent , self.W_enc.t()) + self.b_enc)

        # Rastrear  neurônios mortos durante o treinamento
        if self.training:
            with  torch.no_grad ():
                # Elementos ativos neste  batch ( flag booleana)
                ativos = (z > 0).any(dim =0)
                # Incrementar os que não ativaram , resetar os que ativaram
                self. passos_sem_ativacao += 1
                self. passos_sem_ativacao [ativos] = 0

        # 2) Decodificação (normalizar  W_dec  para evitar aumento artificial de escala)
        # Restringe  norma L2 de cada coluna de W_dec para 1.0
        with torch.no_grad ():
            self.W_dec.data = F.normalize(self.W_dec.data , dim =0)

        x_hat = torch.matmul(z, self.W_dec.t()) + self.b_dec

        # 3) Cálculo de Perda
        loss_rec = F.mse_loss(x_hat , x, reduction= " mean")
        loss_l1   = self.l1_coef * z.sum(dim=-1).mean ()
        loss = loss_rec + loss_l1

        return x_hat , z, loss , loss_rec , loss_l1

    @torch.no_grad ()
    def ressuscitar_neuronios_mortos (self , x_inputs: torch.Tensor , threshold_passos : int = 10000):
        """
        Ressuscita  neurônios que não ativaram  por um threshold de passos
        substituindo  seus pesos com os piores  erros de reconstrução.
        """
        mortos = self. passos_sem_ativacao  > threshold_passos
        n_mortos = mortos. sum().item ()

        if n_mortos == 0:
            return

        # Calcular  erros de reconstrução das entradas atuais
        x_hat , _, _, _, _ = self(x_inputs)
        erros = (x_inputs - x_hat). pow (2).sum(dim =-1)

        # Selecionar os piores  erros para  servir de base para novos neurônios
        _, indices_piores = torch.topk(erros , k= min (n_mortos , len(erros)))

        # Re -inicializar pesos dos neurônios  mortos
        for i, idx_morto  in enumerate(torch.where( mortos)[0]):
            if i >= len ( indices_piores):
                break
            x_exemplo = x_inputs[ indices_piores[i]]

            # Novo peso do decoder = direção do erro normalizada
            erro_vetor = x_exemplo - x_hat[ indices_piores [i]]
            novo_peso = F.normalize(erro_vetor , dim =0)

            self.W_dec.data[:, idx_morto] = novo_peso
            self.W_enc.data[idx_morto , :] = novo_peso * 0.2  # Escala  menor para início
            self.b_enc.data[idx_morto] = 0.0
            self. passos_sem_ativacao [idx_morto] = 0

        print(f"SAE: {n_mortos} neurônios  mortos ressuscitados.")
