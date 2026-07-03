import torch
import torch.nn as nn

class LLMComWeightTying (nn.Module):
    def __init__(self , vocab_size: int , d_model: int):
        super ().__init__ ()
        # 1) Definir a camada de embedding
        self.embedding = nn.Embedding(vocab_size , d_model)

        # 2) Definir a camada de saída (unembedding)
        self.unembedding = nn.Linear(d_model , vocab_size , bias=False)

        # 3) Efetuar o Weight  Tying (compartilhar o mesmo parâmetro físico)
        self.unembedding.weight = self.embedding.weight

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) contendo  IDs de tokens
        emb = self.embedding(x)   # (B, T, d_model)

        # Processamento  simulado por blocos internos
        representacao_final = emb * 1.0

        # Projeta de volta  para logits usando a mesma matriz
        logits = self.unembedding( representacao_final ) # (B, T, vocab_size)
        return  logits
