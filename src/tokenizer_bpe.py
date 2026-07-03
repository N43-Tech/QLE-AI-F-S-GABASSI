from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer


@dataclass(frozen=True)
class TokensEspeciaisBPE:
    pad: int
    bos: int
    eos: int
    unk: int
    usuario: int
    assistente: int


class TokenizadorBPE:
    def __init__(
        self,
        caminho: str | Path = "tokenizer/qle_bpe_2000.json",
    ) -> None:
        self.caminho = Path(caminho)

        if not self.caminho.exists():
            raise FileNotFoundError(
                f"Tokenizador BPE não encontrado: {self.caminho}"
            )

        self._tokenizador = Tokenizer.from_file(
            str(self.caminho)
        )

        ids = {
            "pad": self._tokenizador.token_to_id("<PAD>"),
            "bos": self._tokenizador.token_to_id("<BOS>"),
            "eos": self._tokenizador.token_to_id("<EOS>"),
            "unk": self._tokenizador.token_to_id("<UNK>"),
            "usuario": self._tokenizador.token_to_id("<USER>"),
            "assistente": self._tokenizador.token_to_id("<ASSISTANT>"),
        }

        if any(valor is None for valor in ids.values()):
            raise ValueError(
                "O tokenizador não contém todos os tokens especiais."
            )

        self.especiais = TokensEspeciaisBPE(
            **{chave: int(valor) for chave, valor in ids.items()}
        )
        self.tamanho_vocab = self._tokenizador.get_vocab_size()

    @staticmethod
    def normalizar(texto: str) -> str:
        return unicodedata.normalize("NFC", texto).strip()

    def codificar(
        self,
        texto: str,
        adicionar_bos: bool = False,
        adicionar_eos: bool = False,
    ) -> list[int]:
        texto = self.normalizar(texto)
        ids = self._tokenizador.encode(
            texto,
            add_special_tokens=False,
        ).ids

        if adicionar_bos:
            ids.insert(0, self.especiais.bos)

        if adicionar_eos:
            ids.append(self.especiais.eos)

        return ids

    def codificar_prompt(
        self,
        pergunta: str,
    ) -> list[int]:
        return [
            self.especiais.bos,
            self.especiais.usuario,
            *self.codificar(pergunta),
            self.especiais.assistente,
        ]

    def codificar_dialogo(
        self,
        pergunta: str,
        resposta: str,
    ) -> tuple[list[int], int]:
        pergunta_ids = self.codificar(pergunta)
        resposta_ids = self.codificar(resposta)

        tokens = [
            self.especiais.bos,
            self.especiais.usuario,
            *pergunta_ids,
            self.especiais.assistente,
            *resposta_ids,
            self.especiais.eos,
        ]

        indice_assistente = (
            2 + len(pergunta_ids)
        )

        return tokens, indice_assistente

    def decodificar(
        self,
        ids: Iterable[int],
        ignorar_especiais: bool = True,
    ) -> str:
        valores = [int(token_id) for token_id in ids]

        return self._tokenizador.decode(
            valores,
            skip_special_tokens=ignorar_especiais,
        )
