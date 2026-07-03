from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import DatasetDialogosBPE
from src.model import ConfigModelo, MiniLLM
from src.tokenizer_bpe import TokenizadorBPE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizador",
        type=Path,
        default=Path(
            "tokenizer/qle_bpe_2000.json"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/splits/train.jsonl"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config_qle_bpe_smoke.json"
        ),
    )
    args = parser.parse_args()

    tokenizador = TokenizadorBPE(
        args.tokenizador
    )

    dados_config = json.loads(
        args.config.read_text(
            encoding="utf-8-sig"
        )
    )
    cfg = ConfigModelo(
        **dados_config["modelo"]
    )

    if (
        cfg.tamanho_vocab
        != tokenizador.tamanho_vocab
    ):
        raise ValueError(
            "tamanho_vocab da configuração não corresponde "
            "ao tokenizador."
        )

    dataset = DatasetDialogosBPE(
        args.dataset,
        tokenizador,
        cfg.comprimento_max,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    x, y = next(iter(loader))

    if int(x.max()) >= cfg.tamanho_vocab:
        raise ValueError(
            "O dataset contém ID fora do vocabulário."
        )

    modelo = MiniLLM(cfg)
    logits, perda = modelo(x, y)

    if perda is None or not math.isfinite(
        float(perda.item())
    ):
        raise RuntimeError(
            "A loss do smoke test não é finita."
        )

    primeiro_x, primeiro_y = dataset[0]
    ids_reais = primeiro_x[
        primeiro_x
        != tokenizador.especiais.pad
    ].tolist()

    print("=" * 65)
    print("FASES 4 A 6 VALIDADAS")
    print("=" * 65)
    print(
        f"Vocabulário: "
        f"{tokenizador.tamanho_vocab:,}"
    )
    print(
        f"Exemplos no treino: "
        f"{len(dataset):,}"
    )
    print(
        f"Parâmetros: "
        f"{modelo.contar_parametros():,}"
    )
    print(
        f"Formato dos logits: "
        f"{tuple(logits.shape)}"
    )
    print(
        f"Loss inicial aleatória: "
        f"{float(perda.item()):.4f}"
    )
    print(
        f"Tokens com loss no primeiro exemplo: "
        f"{int((primeiro_y != -100).sum().item())}"
    )
    print(
        "Decodificação do primeiro exemplo:"
    )
    print(
        tokenizador.decodificar(
            ids_reais,
            ignorar_especiais=False,
        )
    )


if __name__ == "__main__":
    main()
