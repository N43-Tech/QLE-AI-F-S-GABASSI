from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MARCADORES_CORROMPIDOS = (
    "Ã¡", "Ã©", "Ãª", "Ã£", "Ã§", "Ã³", "Ã´", "Ãµ",
    "Ãº", "Ã­", "â€™", "â€œ", "â€", "�", "├", "┬", "\x00",
)


def normalizar_categoria(valor: str) -> str:
    valor = unicodedata.normalize("NFKD", valor)
    valor = "".join(
        caractere
        for caractere in valor
        if not unicodedata.combining(caractere)
    )
    valor = re.sub(r"[\s-]+", "_", valor.lower().strip())
    valor = re.sub(r"[^a-z0-9_]", "", valor)

    aliases = {
        "conhecimento_geral": "conhecimentos_gerais",
        "geral": "conhecimentos_gerais",
        "incerteza": "limites",
        "seguranca": "limites",
        "fisica_aplicada": "fisica",
        "eletronica_industrial": "eletronica",
    }
    return aliases.get(valor, valor)


def normalizar_texto(texto: str, nome_ia: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    texto = re.sub(
        r"\bqle\b",
        nome_ia,
        texto,
        flags=re.IGNORECASE,
    )
    return texto


def chave_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )
    return re.sub(r"\s+", " ", texto.lower()).strip()


def ler_jsonl(caminho: Path) -> list[dict[str, Any]]:
    itens: list[dict[str, Any]] = []

    with caminho.open("r", encoding="utf-8-sig") as arquivo:
        for numero_linha, linha in enumerate(arquivo, start=1):
            if not linha.strip():
                continue

            try:
                item = json.loads(linha)
            except json.JSONDecodeError as erro:
                raise ValueError(
                    f"JSON inválido em {caminho}, linha "
                    f"{numero_linha}: {erro}"
                ) from erro

            if not isinstance(item, dict):
                raise ValueError(
                    f"Linha {numero_linha}: era esperado um objeto JSON."
                )

            item["_linha"] = numero_linha
            itens.append(item)

    return itens


def limpar(
    itens: list[dict[str, Any]],
    nome_ia: str,
) -> tuple[list[dict[str, str]], list[str]]:
    limpos: list[dict[str, str]] = []
    erros: list[str] = []
    ids: set[str] = set()
    assinaturas: set[str] = set()

    for item in itens:
        linha = int(item["_linha"])
        identificador = str(item.get("id", "")).strip()
        categoria = normalizar_categoria(
            str(item.get("categoria", ""))
        )
        pergunta = normalizar_texto(
            str(item.get("pergunta", "")),
            nome_ia,
        )
        resposta = normalizar_texto(
            str(item.get("resposta", "")),
            nome_ia,
        )

        if not identificador:
            erros.append(f"Linha {linha}: ID vazio.")
            continue

        if identificador in ids:
            erros.append(
                f"Linha {linha}: ID duplicado '{identificador}'."
            )
            continue

        # Aceita categorias novas como fisica e eletronica, mas rejeita
        # categorias vazias ou com formato inválido.
        if not categoria or not re.fullmatch(r"[a-z0-9_]+", categoria):
            erros.append(
                f"Linha {linha}: categoria inválida '{categoria}'."
            )
            continue

        if len(pergunta) < 3:
            erros.append(f"Linha {linha}: pergunta curta demais.")
            continue

        if len(resposta) < 2:
            erros.append(f"Linha {linha}: resposta curta demais.")
            continue

        marcadores_encontrados = [
            marcador
            for marcador in MARCADORES_CORROMPIDOS
            if marcador in pergunta or marcador in resposta
        ]

        if marcadores_encontrados:
            erros.append(
                f"Linha {linha}: texto corrompido "
                f"{marcadores_encontrados}."
            )
            continue

        base = (
            f"{chave_texto(pergunta)}\n"
            f"{chave_texto(resposta)}"
        )
        assinatura = hashlib.sha256(
            base.encode("utf-8")
        ).hexdigest()

        if assinatura in assinaturas:
            erros.append(
                f"Linha {linha}: duplicado por conteúdo."
            )
            continue

        ids.add(identificador)
        assinaturas.add(assinatura)

        limpos.append(
            {
                "id": identificador,
                "categoria": categoria,
                "pergunta": pergunta,
                "resposta": resposta,
            }
        )

    return limpos, erros


def dividir(
    itens: list[dict[str, str]],
    seed: int,
    fracao_validacao: float,
    fracao_teste: float,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    if fracao_validacao < 0 or fracao_teste < 0:
        raise ValueError("As frações não podem ser negativas.")

    if fracao_validacao + fracao_teste >= 0.5:
        raise ValueError(
            "A soma de validação e teste deve ser menor que 0.5."
        )

    grupos_por_categoria: dict[
        str,
        dict[str, list[dict[str, str]]],
    ] = defaultdict(lambda: defaultdict(list))

    for item in itens:
        grupos_por_categoria[
            item["categoria"]
        ][chave_texto(item["pergunta"])].append(item)

    gerador = random.Random(seed)
    treino: list[dict[str, str]] = []
    validacao: list[dict[str, str]] = []
    teste: list[dict[str, str]] = []

    for categoria, grupos_dict in sorted(
        grupos_por_categoria.items()
    ):
        grupos = list(grupos_dict.values())
        gerador.shuffle(grupos)
        quantidade = len(grupos)

        if quantidade < 3:
            for grupo in grupos:
                treino.extend(grupo)
            continue

        quantidade_validacao = max(
            1,
            round(quantidade * fracao_validacao),
        )
        quantidade_teste = max(
            1,
            round(quantidade * fracao_teste),
        )

        if quantidade_validacao + quantidade_teste >= quantidade:
            quantidade_validacao = 1
            quantidade_teste = 1

        quantidade_treino = (
            quantidade
            - quantidade_validacao
            - quantidade_teste
        )

        for grupo in grupos[:quantidade_treino]:
            treino.extend(grupo)

        inicio_validacao = quantidade_treino
        fim_validacao = (
            inicio_validacao + quantidade_validacao
        )

        for grupo in grupos[
            inicio_validacao:fim_validacao
        ]:
            validacao.extend(grupo)

        for grupo in grupos[fim_validacao:]:
            teste.extend(grupo)

    gerador.shuffle(treino)
    gerador.shuffle(validacao)
    gerador.shuffle(teste)

    return treino, validacao, teste


def escrever_jsonl(
    caminho: Path,
    itens: list[dict[str, str]],
) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with caminho.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as arquivo:
        for item in itens:
            arquivo.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Limpa, valida, remove duplicados e divide "
            "o corpus da QLE."
        )
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=Path("data/raw/qle_dataset.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nome-ia", type=str, default="QLE")
    parser.add_argument(
        "--validacao",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--teste",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--estrito",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.entrada.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {args.entrada.resolve()}"
        )

    exemplos_brutos = ler_jsonl(args.entrada)
    exemplos_limpos, erros = limpar(
        exemplos_brutos,
        args.nome_ia,
    )

    if not exemplos_limpos:
        raise ValueError("Nenhum exemplo válido restou.")

    if args.estrito and erros:
        primeiros_erros = "\n".join(erros[:50])
        raise ValueError(
            f"Foram encontrados {len(erros)} erros:\n"
            f"{primeiros_erros}"
        )

    treino, validacao, teste = dividir(
        exemplos_limpos,
        args.seed,
        args.validacao,
        args.teste,
    )

    escrever_jsonl(
        Path("data/processed/qle_limpo.jsonl"),
        exemplos_limpos,
    )
    escrever_jsonl(
        Path("data/splits/train.jsonl"),
        treino,
    )
    escrever_jsonl(
        Path("data/splits/validation.jsonl"),
        validacao,
    )
    escrever_jsonl(
        Path("data/splits/test.jsonl"),
        teste,
    )

    contagem = Counter(
        item["categoria"]
        for item in exemplos_limpos
    )

    relatorio = {
        "brutos": len(exemplos_brutos),
        "validos": len(exemplos_limpos),
        "removidos": len(erros),
        "categorias": dict(sorted(contagem.items())),
        "splits": {
            "train": len(treino),
            "validation": len(validacao),
            "test": len(teste),
        },
        "erros": erros,
    }

    caminho_relatorio = Path(
        "data/processed/relatorio_corpus.json"
    )
    caminho_relatorio.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    caminho_relatorio.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 65)
    print("CORPUS QLE PREPARADO")
    print("=" * 65)
    print(f"Exemplos brutos: {len(exemplos_brutos):,}")
    print(f"Exemplos válidos: {len(exemplos_limpos):,}")
    print(f"Removidos: {len(erros):,}")
    print(f"Treino: {len(treino):,}")
    print(f"Validação: {len(validacao):,}")
    print(f"Teste: {len(teste):,}")
    print("\nCategorias:")

    for categoria in sorted(contagem):
        print(
            f"- {categoria:<24} "
            f"{contagem[categoria]:>6}"
        )

    print(
        f"\nRelatório: {caminho_relatorio.resolve()}"
    )


if __name__ == "__main__":
    main()
