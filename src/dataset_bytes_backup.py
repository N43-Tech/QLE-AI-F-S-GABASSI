from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DatasetCausal(Dataset):
    def __init__(self, tokens: list[int], comprimento: int) -> None:
        if len(tokens) <= comprimento:
            raise ValueError(
                "O corpus é curto demais. Adicione mais texto ou reduza comprimento_max."
            )
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.comprimento = comprimento

    def __len__(self) -> int:
        return len(self.tokens) - self.comprimento

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, torch.Tensor]:
        trecho = self.tokens[indice : indice + self.comprimento + 1]
        return trecho[:-1], trecho[1:]
