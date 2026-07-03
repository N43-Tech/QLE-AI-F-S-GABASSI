from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.model import ConfigModelo, MiniLLM
from src.tokenizer_bpe import TokenizadorBPE


def criar_config(
    tamanho_vocab: int,
    passos: int,
) -> dict:
    return {
        "modelo": {
            "tamanho_vocab": tamanho_vocab,
            "d_model": 192,
            "n_camadas": 6,
            "n_cabecas": 6,
            "n_cabecas_kv": 2,
            "d_ff": 576,
            "comprimento_max": 256,
            "dropout": 0.1,
            "rope_theta": 10000.0,
        },
        "treino": {
            "batch_size": 4,
            "passos": passos,
            "warmup_passos": min(1000, max(10, passos // 10)),
            "lr_max": 0.0003,
            "lr_min": 0.00001,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "log_intervalo": 10 if passos <= 100 else 25,
            "validar_cada": 25 if passos <= 100 else 250,
            "salvar_cada": 25 if passos <= 100 else 250,
            "lotes_validacao": 30,
            "acumular_gradientes": 4,
            "num_workers": 0,
        },
    }


def escrever(
    caminho: Path,
    dados: dict,
) -> None:
    caminho.write_text(
        json.dumps(
            dados,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizador",
        type=Path,
        default=Path(
            "tokenizer/qle_bpe_2000.json"
        ),
    )
    args = parser.parse_args()

    tokenizador = TokenizadorBPE(
        args.tokenizador
    )

    config_smoke = criar_config(
        tokenizador.tamanho_vocab,
        passos=100,
    )
    config_principal = criar_config(
        tokenizador.tamanho_vocab,
        passos=20000,
    )

    caminho_smoke = Path(
        "config_qle_bpe_smoke.json"
    )
    caminho_principal = Path(
        "config_qle_bpe_3m.json"
    )

    escrever(
        caminho_smoke,
        config_smoke,
    )
    escrever(
        caminho_principal,
        config_principal,
    )

    cfg = ConfigModelo(
        **config_principal["modelo"]
    )
    modelo = MiniLLM(cfg)

    print("=" * 65)
    print("CONFIGURAÇÕES QLE BPE CRIADAS")
    print("=" * 65)
    print(
        f"Vocabulário real: "
        f"{tokenizador.tamanho_vocab:,}"
    )
    print(
        f"Parâmetros do modelo: "
        f"{modelo.contar_parametros():,}"
    )
    print(
        f"Configuração de teste: "
        f"{caminho_smoke.resolve()}"
    )
    print(
        f"Configuração principal: "
        f"{caminho_principal.resolve()}"
    )


if __name__ == "__main__":
    main()
