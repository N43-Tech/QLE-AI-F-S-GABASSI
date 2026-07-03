from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.dataset import DatasetDialogosBPE
from src.generate import (
    gerar_resposta,
    carregar_modelo_e_tokenizador,
    escolher_dispositivo,
)
from src.train import (
    avaliar_modelo,
    perplexidade_segura,
)


def normalizar_texto(
    texto: str,
) -> str:
    texto = unicodedata.normalize(
        "NFKD",
        texto,
    )

    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(
            caractere
        )
    )

    texto = texto.lower()
    texto = re.sub(
        r"[^\w\s]",
        " ",
        texto,
    )
    texto = re.sub(
        r"\s+",
        " ",
        texto,
    ).strip()

    return texto


def f1_palavras(
    esperado: str,
    gerado: str,
) -> float:
    esperado_tokens = normalizar_texto(
        esperado
    ).split()
    gerado_tokens = normalizar_texto(
        gerado
    ).split()

    if not esperado_tokens and not gerado_tokens:
        return 1.0

    if not esperado_tokens or not gerado_tokens:
        return 0.0

    contagem_esperado: dict[str, int] = {}
    contagem_gerado: dict[str, int] = {}

    for token in esperado_tokens:
        contagem_esperado[token] = (
            contagem_esperado.get(
                token,
                0,
            )
            + 1
        )

    for token in gerado_tokens:
        contagem_gerado[token] = (
            contagem_gerado.get(
                token,
                0,
            )
            + 1
        )

    comuns = sum(
        min(
            quantidade,
            contagem_gerado.get(
                token,
                0,
            ),
        )
        for token, quantidade
        in contagem_esperado.items()
    )

    if comuns == 0:
        return 0.0

    precisao = (
        comuns / len(gerado_tokens)
    )
    revocacao = (
        comuns / len(esperado_tokens)
    )

    return (
        2
        * precisao
        * revocacao
        / (precisao + revocacao)
    )


def taxa_repeticao(
    texto: str,
) -> float:
    tokens = normalizar_texto(
        texto
    ).split()

    if len(tokens) < 2:
        return 0.0

    return 1.0 - (
        len(set(tokens)) / len(tokens)
    )


def carregar_perguntas(
    caminho: Path,
    limite: int | None,
) -> list[dict[str, str]]:
    exemplos: list[dict[str, str]] = []

    with caminho.open(
        "r",
        encoding="utf-8-sig",
    ) as arquivo:
        for numero, linha in enumerate(
            arquivo,
            start=1,
        ):
            if not linha.strip():
                continue

            item = json.loads(linha)

            pergunta = str(
                item.get("pergunta", "")
            ).strip()
            resposta = str(
                item.get("resposta", "")
            ).strip()

            if not pergunta:
                raise ValueError(
                    f"Linha {numero}: pergunta vazia."
                )

            exemplos.append(
                {
                    "id": str(
                        item.get("id", numero)
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

            if (
                limite is not None
                and len(exemplos) >= limite
            ):
                break

    return exemplos


def agora_utc_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")


def criar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia loss, perplexidade e respostas geradas pela QLE."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/qle_bpe_melhor.pt"
        ),
    )

    parser.add_argument(
        "--tokenizador",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/splits/test.jsonl"
        ),
    )

    parser.add_argument(
        "--perguntas",
        type=Path,
        default=None,
        help=(
            "JSONL opcional. Quando omitido, usa as perguntas "
            "do próprio dataset de teste."
        ),
    )

    parser.add_argument(
        "--saida",
        type=Path,
        default=Path(
            "avaliacao/resultados_qle.json"
        ),
    )

    parser.add_argument(
        "--limite",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--lotes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--temperatura",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--penalidade-repeticao",
        type=float,
        default=1.10,
    )

    parser.add_argument(
        "--no-repeat-ngram",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--dispositivo",
        choices=(
            "auto",
            "cpu",
            "cuda",
            "mps",
        ),
        default="auto",
    )

    return parser.parse_args()


def main() -> None:
    args = criar_argumentos()

    dispositivo = escolher_dispositivo(
        args.dispositivo
    )

    (
        modelo,
        cfg,
        tokenizador,
        checkpoint,
        tokenizador_path,
    ) = carregar_modelo_e_tokenizador(
        caminho_checkpoint=(
            args.checkpoint
        ),
        caminho_tokenizador=(
            args.tokenizador
        ),
        dispositivo=dispositivo,
    )

    dataset = DatasetDialogosBPE(
        args.dataset,
        tokenizador,
        cfg.comprimento_max,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    test_loss, lotes = avaliar_modelo(
        modelo,
        loader,
        dispositivo,
        args.lotes,
    )

    caminho_perguntas = (
        args.perguntas
        if args.perguntas is not None
        else args.dataset
    )

    perguntas = carregar_perguntas(
        caminho_perguntas,
        args.limite,
    )

    resultados: list[dict[str, Any]] = []

    for indice, item in enumerate(
        perguntas,
        start=1,
    ):
        resposta, eos, quantidade = (
            gerar_resposta(
                modelo=modelo,
                cfg=cfg,
                tokenizador=tokenizador,
                mensagem=item["pergunta"],
                quantidade_tokens=(
                    args.tokens
                ),
                temperatura=(
                    args.temperatura
                ),
                top_k=args.top_k,
                top_p=args.top_p,
                penalidade_repeticao=(
                    args.penalidade_repeticao
                ),
                no_repeat_ngram=(
                    args.no_repeat_ngram
                ),
            )
        )

        esperado = item["resposta"]

        resultado = {
            **item,
            "resposta_gerada": resposta,
            "atingiu_eos": eos,
            "tokens_gerados": quantidade,
            "resposta_vazia": (
                not resposta.strip()
            ),
            "exact_match": (
                normalizar_texto(resposta)
                == normalizar_texto(
                    esperado
                )
                if esperado
                else None
            ),
            "f1_palavras": (
                f1_palavras(
                    esperado,
                    resposta,
                )
                if esperado
                else None
            ),
            "taxa_repeticao": (
                taxa_repeticao(
                    resposta
                )
            ),
        }

        resultados.append(resultado)

        print(
            f"[{indice:>3}/{len(perguntas)}] "
            f"{item['id']} | eos={eos} | "
            f"tokens={quantidade}"
        )

    total = len(resultados)

    exact_validos = [
        item["exact_match"]
        for item in resultados
        if item["exact_match"] is not None
    ]

    f1_validos = [
        float(item["f1_palavras"])
        for item in resultados
        if item["f1_palavras"] is not None
    ]

    resumo = {
        "gerado_em_utc": agora_utc_iso(),
        "checkpoint": str(
            args.checkpoint.resolve()
        ),
        "passo_checkpoint": checkpoint.get(
            "passo_atual"
        ),
        "tokenizador": str(
            tokenizador_path.resolve()
        ),
        "dataset": str(
            args.dataset.resolve()
        ),
        "test_loss": test_loss,
        "test_perplexity": (
            perplexidade_segura(
                test_loss
            )
        ),
        "lotes_avaliados": lotes,
        "respostas_avaliadas": total,
        "taxa_eos": (
            sum(
                bool(item["atingiu_eos"])
                for item in resultados
            )
            / total
            if total
            else 0.0
        ),
        "taxa_resposta_vazia": (
            sum(
                bool(item["resposta_vazia"])
                for item in resultados
            )
            / total
            if total
            else 0.0
        ),
        "exact_match": (
            sum(exact_validos)
            / len(exact_validos)
            if exact_validos
            else None
        ),
        "f1_palavras_medio": (
            sum(f1_validos)
            / len(f1_validos)
            if f1_validos
            else None
        ),
        "repeticao_media": (
            sum(
                float(
                    item[
                        "taxa_repeticao"
                    ]
                )
                for item in resultados
            )
            / total
            if total
            else 0.0
        ),
    }

    saida = {
        "resumo": resumo,
        "resultados": resultados,
    }

    args.saida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.saida.write_text(
        json.dumps(
            saida,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("AVALIAÇÃO CONCLUÍDA")
    print("=" * 72)
    print(
        f"test_loss: {test_loss:.4f}"
    )
    print(
        "test_ppl: "
        f"{perplexidade_segura(test_loss):.2f}"
    )
    print(
        f"taxa_eos: "
        f"{resumo['taxa_eos']:.2%}"
    )
    print(
        f"f1 médio: "
        f"{resumo['f1_palavras_medio']}"
    )
    print(
        f"Resultado: "
        f"{args.saida.resolve()}"
    )


if __name__ == "__main__":
    main()
