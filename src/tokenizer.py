from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TokensEspeciais:
    pad: int = 256
    bos: int = 257
    eos: int = 258
    unk: int = 259


class TokenizadorBytes:
    """Tokenizador UTF-8 reversível, sem vocabulário externo.

    Cada byte ocupa um ID entre 0 e 255. Quatro IDs extras são
    reservados para PAD, BOS, EOS e UNK. É menos eficiente que BPE,
    mas permite treinar o primeiro modelo sem depender de outro pacote.
    """

    def __init__(self) -> None:
        self.especiais = TokensEspeciais()
        self.tamanho_vocab = 260

    def codificar(
        self,
        texto: str,
        adicionar_bos: bool = False,
        adicionar_eos: bool = False,
    ) -> list[int]:
        ids = list(texto.encode("utf-8", errors="replace"))
        if adicionar_bos:
            ids.insert(0, self.especiais.bos)
        if adicionar_eos:
            ids.append(self.especiais.eos)
        return ids

    def decodificar(self, ids: Iterable[int], ignorar_especiais: bool = True) -> str:
        bytes_validos: list[int] = []
        for token_id in ids:
            valor = int(token_id)
            if 0 <= valor <= 255:
                bytes_validos.append(valor)
            elif not ignorar_especiais:
                mapa = {
                    self.especiais.pad: "<PAD>",
                    self.especiais.bos: "<BOS>",
                    self.especiais.eos: "<EOS>",
                    self.especiais.unk: "<UNK>",
                }
                bytes_validos.extend(mapa.get(valor, "<UNK>").encode("utf-8"))
        return bytes(bytes_validos).decode("utf-8", errors="replace")

    def salvar_tokens(self, texto: str, caminho: str | Path) -> int:
        destino = Path(caminho)
        destino.parent.mkdir(parents=True, exist_ok=True)
        ids = self.codificar(texto, adicionar_bos=True, adicionar_eos=True)
        import numpy as np

        np.asarray(ids, dtype=np.uint16).tofile(destino)
        return len(ids)
