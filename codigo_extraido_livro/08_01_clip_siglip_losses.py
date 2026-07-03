import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    """
    Perda contrastiva  simetrica utilizada no CLIP ( InfoNCE).
    """
    def __init__(self , temperatura_inicial : float = 0.07):
        super ().__init__ ()
        # Registrar  temperatura como parametro a ser aprendido
        self.t_log = nn.Parameter(torch.tensor( torch.log(torch.tensor( temperatura_inicial ))))

    def forward(self , embeddings_imagem : torch.Tensor , embeddings_texto : torch.Tensor) -> torch.Tensor:
        # 1) Normalizar  embeddings
        img_norm = F.normalize(embeddings_imagem , p =2, dim=-1)
        txt_norm = F.normalize(embeddings_texto , p =2, dim=-1)

        # 2) Calcular  matriz de similaridades cosseno
        # (B, d) @ (d, B) -> (B, B)
        sim_matrix = torch.matmul(img_norm , txt_norm.t())

        # 3) Aplicar  temperatura aprendida
        temperatura = torch.exp(self.t_log)
        logits = sim_matrix / temperatura

        # 4) Alvos da diagonal (index correspondente da imagem ao texto)
        B = logits.shape [0]
        targets = torch.arange(B, device=logits.device)

        # 5) Cross  entropy em ambas as direcoes ( imagem ->texto e texto ->imagem)
        loss_img = F.cross_entropy(logits , targets , reduction='mean ')
        loss_txt = F.cross_entropy(logits.t(), targets , reduction='mean ')

        return (loss_img + loss_txt) / 2.0


class SigLIPLoss(nn.Module):
    """
    Perda baseada em Sigmoid  proposta no SigLIP.
    """
    def __init__(self , escala_inicial: float = 10.0 , bias_inicial: float =  -10.0):
        super ().__init__ ()
        self.t_log = nn.Parameter(torch.log(torch.tensor(escala_inicial )))
        self.bias = nn.Parameter(torch.tensor( bias_inicial))

    def forward(self , embeddings_imagem : torch.Tensor , embeddings_texto : torch.Tensor) -> torch.Tensor:
        img_norm = F.normalize(embeddings_imagem , p =2, dim=-1)
        txt_norm = F.normalize(embeddings_texto , p =2, dim=-1)

        # Matriz de similaridades (B, B)
        sim_matrix = torch.matmul(img_norm , txt_norm.t())

        B = sim_matrix.shape [0]
        escala = torch.exp(self.t_log)

        # logits = (similaridade - bias) * escala
        logits = (sim_matrix - self.bias) * escala

        # Criar  rotulos: +1 na diagonal (pares corretos) e -1 fora
        labels = 2.0 * torch.eye(B, device= sim_matrix.device) - 1.0

        # Perda: -log_sigmoid(labels * logits)
        loss = -F.logsigmoid(labels * logits)

        # Normalizar  perda separadamente  para positivos e negativos
        loss_pos = loss.diag ().mean ()
        # Mascara  para excluir a diagonal na contagem dos negativos
        mask_neg = (1.0 - torch.eye(B, device= sim_matrix.device))
        loss_neg = (loss * mask_neg). sum() / (B * ( B - 1))

        return  loss_pos + loss_neg
