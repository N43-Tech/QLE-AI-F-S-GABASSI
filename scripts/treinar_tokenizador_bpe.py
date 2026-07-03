from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


TOKENS_ESPECIAIS = (
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
    "<USER>",
    "<ASSISTANT>",
)


def normalizar(texto: str) -> str:
    return unicodedata.normalize("NFC", texto).strip()


def iterar_textos(caminho: Path) -> Iterator[str]:
    with caminho.open("r", encoding="utf-8-sig") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            if not linha.strip():
                continue

            try:
                item = json.loads(linha)
            except json.JSONDecodeError as erro:
                raise ValueError(
                    f"JSON inválido em {caminho}, linha {numero}: {erro}"
                ) from erro

            pergunta = normalizar(str(item.get("pergunta", "")))
            resposta = normalizar(str(item.get("resposta", "")))

            if not pergunta or not resposta:
                raise ValueError(
                    f"Linha {numero}: pergunta ou resposta vazia."
                )

            yield f"{pergunta}\n{resposta}"


def carregar_amostras(caminho: Path, limite: int = 100) -> list[str]:
    return list(iterar_textos(caminho))[:limite]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina um tokenizador BPE da QLE inteiramente do zero."
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=Path("data/splits/train.jsonl"),
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("tokenizer/qle_bpe_2000.json"),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    if not args.entrada.exists():
        raise FileNotFoundError(
            f"Arquivo de treino não encontrado: {args.entrada}"
        )

    if args.vocab_size < 512:
        raise ValueError("--vocab-size deve ser pelo menos 512.")

    if args.min_frequency < 1:
        raise ValueError("--min-frequency deve ser pelo menos 1.")

    tokenizador = Tokenizer(
        BPE(
            unk_token="<UNK>",
        )
    )
    tokenizador.normalizer = NFC()
    tokenizador.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizador.decoder = ByteLevelDecoder()

    treinador = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=list(TOKENS_ESPECIAIS),
        initial_alphabet=ByteLevel.alphabet(),
        max_token_length=32,
    )

    tokenizador.train_from_iterator(
        iterar_textos(args.entrada),
        trainer=treinador,
    )

    args.saida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tokenizador.save(str(args.saida))

    ids_especiais = {
        token: tokenizador.token_to_id(token)
        for token in TOKENS_ESPECIAIS
    }

    if any(valor is None for valor in ids_especiais.values()):
        raise RuntimeError(
            "Um ou mais tokens especiais não foram registrados."
        )

    amostras = carregar_amostras(args.entrada)
    proporcoes: list[float] = []

    for texto in amostras:
        codificacao = tokenizador.encode(
            texto,
            add_special_tokens=False,
        )
        quantidade_tokens = max(1, len(codificacao.ids))
        proporcoes.append(len(texto) / quantidade_tokens)

    metadados = {
        "arquivo": str(args.saida),
        "tipo": "BPE ByteLevel",
        "vocab_solicitado": args.vocab_size,
        "vocab_real": tokenizador.get_vocab_size(),
        "min_frequency": args.min_frequency,
        "tokens_especiais": ids_especiais,
        "amostras_avaliadas": len(amostras),
        "caracteres_por_token_media": (
            statistics.mean(proporcoes) if proporcoes else None
        ),
    }

    caminho_meta = args.saida.with_name(
        args.saida.stem + "_meta.json"
    )
    caminho_meta.write_text(
        json.dumps(
            metadados,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 65)
    print("TOKENIZADOR BPE DA QLE TREINADO")
    print("=" * 65)
    print(f"Entrada: {args.entrada.resolve()}")
    print(f"Saída: {args.saida.resolve()}")
    print(f"Vocabulário solicitado: {args.vocab_size:,}")
    print(
        f"Vocabulário real: "
        f"{tokenizador.get_vocab_size():,}"
    )
    print("Tokens especiais:")

    for token, token_id in ids_especiais.items():
        print(f"- {token:<12} ID {token_id}")

    if proporcoes:
        print(
            "Média de caracteres por token: "
            f"{statistics.mean(proporcoes):.2f}"
        )

    print(f"Metadados: {caminho_meta.resolve()}")


if __name__ == "__main__":
    main()
