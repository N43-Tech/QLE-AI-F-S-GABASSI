from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.tokenizer_bpe import TokenizadorBPE


class DatasetCausal(Dataset):
    """
    Dataset causal antigo, mantido apenas para compatibilidade.

    Para o treinamento BPE supervisionado da QLE, utilize
    DatasetDialogosBPE.
    """

    def __init__(
        self,
        tokens: list[int],
        comprimento: int,
    ) -> None:
        if comprimento < 1:
            raise ValueError(
                "O comprimento precisa ser maior que zero."
            )

        if len(tokens) <= comprimento:
            raise ValueError(
                "O corpus é curto demais para o comprimento informado."
            )

        self.tokens = torch.tensor(
            tokens,
            dtype=torch.long,
        )
        self.comprimento = comprimento

    def __len__(self) -> int:
        return len(self.tokens) - self.comprimento

    def __getitem__(
        self,
        indice: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trecho = self.tokens[
            indice : indice + self.comprimento + 1
        ]

        return trecho[:-1], trecho[1:]


class DatasetDialogosBPE(Dataset):
    """
    Dataset supervisionado de perguntas e respostas para a QLE.

    Formato conceitual de cada exemplo:

        <BOS><USER>pergunta<ASSISTANT>resposta<EOS>

    A loss é calculada somente sobre:

        resposta<EOS>

    Os tokens da pergunta, dos marcadores e do preenchimento recebem
    alvo -100 e são ignorados pelo CrossEntropyLoss.
    """

    CAMPOS_OBRIGATORIOS = {
        "pergunta",
        "resposta",
    }

    def __init__(
        self,
        caminho_jsonl: str | Path,
        tokenizador: TokenizadorBPE,
        comprimento: int,
        *,
        minimo_tokens_pergunta: int = 8,
    ) -> None:
        self.caminho = Path(caminho_jsonl)
        self.tokenizador = tokenizador
        self.comprimento = int(comprimento)
        self.minimo_tokens_pergunta = int(
            minimo_tokens_pergunta
        )

        if not self.caminho.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado: {self.caminho.resolve()}"
            )

        if self.caminho.stat().st_size == 0:
            raise ValueError(
                f"O dataset está vazio: {self.caminho.resolve()}"
            )

        if self.comprimento < 32:
            raise ValueError(
                "O comprimento máximo precisa ser pelo menos 32."
            )

        if self.minimo_tokens_pergunta < 1:
            raise ValueError(
                "minimo_tokens_pergunta precisa ser pelo menos 1."
            )

        self.exemplos = self._carregar_jsonl()

        if not self.exemplos:
            raise ValueError(
                "Nenhum exemplo válido foi encontrado no dataset."
            )

        self.itens = [
            self._preparar_exemplo(exemplo)
            for exemplo in self.exemplos
        ]

    def _carregar_jsonl(
        self,
    ) -> list[dict[str, str]]:
        exemplos: list[dict[str, str]] = []

        with self.caminho.open(
            "r",
            encoding="utf-8-sig",
        ) as arquivo:
            for numero_linha, linha in enumerate(
                arquivo,
                start=1,
            ):
                if not linha.strip():
                    continue

                try:
                    item: Any = json.loads(linha)
                except json.JSONDecodeError as erro:
                    raise ValueError(
                        f"JSON inválido em {self.caminho}, "
                        f"linha {numero_linha}, "
                        f"coluna {erro.colno}: {erro.msg}"
                    ) from erro

                if not isinstance(item, dict):
                    raise ValueError(
                        f"Linha {numero_linha}: "
                        "era esperado um objeto JSON."
                    )

                campos_ausentes = (
                    self.CAMPOS_OBRIGATORIOS
                    - item.keys()
                )

                if campos_ausentes:
                    raise ValueError(
                        f"Linha {numero_linha}: faltam os campos "
                        f"{sorted(campos_ausentes)}."
                    )

                pergunta = str(
                    item.get("pergunta", "")
                ).strip()
                resposta = str(
                    item.get("resposta", "")
                ).strip()

                if not pergunta:
                    raise ValueError(
                        f"Linha {numero_linha}: pergunta vazia."
                    )

                if not resposta:
                    raise ValueError(
                        f"Linha {numero_linha}: resposta vazia."
                    )

                exemplos.append(
                    {
                        "id": str(
                            item.get(
                                "id",
                                numero_linha,
                            )
                        ),
                        "categoria": str(
                            item.get(
                                "categoria",
                                "sem_categoria",
                            )
                        ),
                        "pergunta": pergunta,
                        "resposta": resposta,
                    }
                )

        return exemplos

    def _ajustar_ao_contexto(
        self,
        pergunta_ids: list[int],
        resposta_ids: list[int],
    ) -> tuple[list[int], int]:
        """
        Ajusta pergunta e resposta ao contexto máximo.

        A sequência completa possui comprimento + 1 tokens porque,
        depois, ela é deslocada para formar entrada e alvos.
        """

        especiais = self.tokenizador.especiais
        limite_total = self.comprimento + 1

        # Tokens fixos:
        # <BOS>, <USER>, <ASSISTANT>, <EOS>
        quantidade_tokens_fixos = 4
        orcamento_conteudo = (
            limite_total - quantidade_tokens_fixos
        )

        if orcamento_conteudo < 2:
            raise ValueError(
                "O contexto é pequeno demais para montar um diálogo."
            )

        # Preserva espaço mínimo para a pergunta e prioriza a resposta,
        # pois é sobre ela que a loss será calculada.
        reserva_pergunta = min(
            self.minimo_tokens_pergunta,
            max(1, len(pergunta_ids)),
            max(1, orcamento_conteudo - 1),
        )

        maximo_resposta = max(
            1,
            orcamento_conteudo - reserva_pergunta,
        )

        if len(resposta_ids) > maximo_resposta:
            resposta_ids = resposta_ids[
                :maximo_resposta
            ]

        espaco_restante_pergunta = max(
            0,
            orcamento_conteudo - len(resposta_ids),
        )

        if espaco_restante_pergunta == 0:
            pergunta_ids = []
        elif len(pergunta_ids) > espaco_restante_pergunta:
            # Mantém o final da pergunta, onde frequentemente está a
            # informação mais específica.
            pergunta_ids = pergunta_ids[
                -espaco_restante_pergunta:
            ]

        tokens = [
            especiais.bos,
            especiais.usuario,
            *pergunta_ids,
            especiais.assistente,
            *resposta_ids,
            especiais.eos,
        ]

        if len(tokens) > limite_total:
            raise RuntimeError(
                "A sequência excedeu o contexto após o corte."
            )

        # Índice do token <ASSISTANT> na sequência de entrada.
        indice_assistente = (
            2 + len(pergunta_ids)
        )

        return tokens, indice_assistente

    def _preparar_exemplo(
        self,
        item: dict[str, str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pergunta_ids = self.tokenizador.codificar(
            item["pergunta"]
        )
        resposta_ids = self.tokenizador.codificar(
            item["resposta"]
        )

        tokens, indice_assistente = (
            self._ajustar_ao_contexto(
                pergunta_ids,
                resposta_ids,
            )
        )

        entrada = list(tokens[:-1])
        alvos = list(tokens[1:])

        # O alvo na posição do <ASSISTANT> da entrada já é o primeiro
        # token da resposta. Portanto, mascaramos somente as posições
        # anteriores.
        limite_mascara = min(
            indice_assistente,
            len(alvos),
        )

        for indice in range(limite_mascara):
            alvos[indice] = -100

        quantidade_padding = (
            self.comprimento - len(entrada)
        )

        if quantidade_padding < 0:
            raise RuntimeError(
                "A entrada ultrapassou o comprimento máximo."
            )

        if quantidade_padding:
            entrada.extend(
                [self.tokenizador.especiais.pad]
                * quantidade_padding
            )
            alvos.extend(
                [-100] * quantidade_padding
            )

        if len(entrada) != self.comprimento:
            raise RuntimeError(
                "A entrada final possui tamanho incorreto."
            )

        if len(alvos) != self.comprimento:
            raise RuntimeError(
                "Os alvos finais possuem tamanho incorreto."
            )

        if not any(
            alvo != -100
            for alvo in alvos
        ):
            raise RuntimeError(
                f"O exemplo '{item['id']}' ficou sem tokens de loss."
            )

        entrada_tensor = torch.tensor(
            entrada,
            dtype=torch.long,
        )
        alvos_tensor = torch.tensor(
            alvos,
            dtype=torch.long,
        )

        if int(entrada_tensor.min().item()) < 0:
            raise RuntimeError(
                f"O exemplo '{item['id']}' contém ID negativo na entrada."
            )

        if (
            int(entrada_tensor.max().item())
            >= self.tokenizador.tamanho_vocab
        ):
            raise RuntimeError(
                f"O exemplo '{item['id']}' contém ID fora "
                "do vocabulário."
            )

        return entrada_tensor, alvos_tensor

    def __len__(self) -> int:
        return len(self.itens)

    def __getitem__(
        self,
        indice: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.itens[indice]

    def estatisticas(self) -> dict[str, int | float]:
        tokens_com_loss = sum(
            int(
                (alvos != -100)
                .sum()
                .item()
            )
            for _, alvos in self.itens
        )

        total_posicoes = (
            len(self.itens)
            * self.comprimento
        )

        return {
            "exemplos": len(self.itens),
            "comprimento": self.comprimento,
            "tokens_com_loss": tokens_com_loss,
            "media_tokens_com_loss": (
                tokens_com_loss / len(self.itens)
            ),
            "percentual_posicoes_com_loss": (
                tokens_com_loss
                / total_posicoes
                * 100
            ),
        }
