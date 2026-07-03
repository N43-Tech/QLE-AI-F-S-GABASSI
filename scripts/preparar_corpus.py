from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CAMPOS_OBRIGATORIOS = {
    "id",
    "categoria",
    "pergunta",
    "resposta",
}

MARCADORES_DE_CODIFICACAO_CORROMPIDA = (
    "Ã¡",
    "Ã©",
    "Ãª",
    "Ã£",
    "Ã§",
    "Ã³",
    "Ãô",
    "Ã´",
    "Ãõ",
    "Ãµ",
    "Ãú",
    "Ãº",
    "Ãí",
    "Ã­",
    "â€™",
    "â€œ",
    "â€",
    "�",
    "├",
    "┬",
    "\x00",
)

ALIASES_DE_CATEGORIA = {
    "automacao_industrial": "automacao",
    "conhecimento_geral": "conhecimentos_gerais",
    "conhecimentos_geral": "conhecimentos_gerais",
    "geral": "conhecimentos_gerais",
    "incerteza": "limites",
    "seguranca": "limites",
    "limites_e_incerteza": "limites",
    "fisica_aplicada": "fisica",
    "eletronica_industrial": "eletronica",
    "lingua_portuguesa": "portugues",
    "programacao_de_computadores": "programacao",
}

PADRAO_CATEGORIA = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class Problema:
    linha: int
    codigo: str
    mensagem: str
    id: str | None = None


@dataclass(frozen=True)
class Registro:
    id: str
    categoria: str
    pergunta: str
    resposta: str
    linha_origem: int

    def para_saida(self) -> dict[str, str]:
        return {
            "id": self.id,
            "categoria": self.categoria,
            "pergunta": self.pergunta,
            "resposta": self.resposta,
        }


class ErroCorpus(RuntimeError):
    """Erro controlado do pipeline de preparação do corpus."""


def agora_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()

    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)

    return digest.hexdigest()


def normalizar_categoria(valor: str) -> str:
    valor = unicodedata.normalize("NFKD", valor)

    valor = "".join(
        caractere
        for caractere in valor
        if not unicodedata.combining(caractere)
    )

    valor = valor.lower().strip()
    valor = re.sub(r"[\s\-./]+", "_", valor)
    valor = re.sub(r"[^a-z0-9_]", "", valor)
    valor = re.sub(r"_+", "_", valor).strip("_")

    return ALIASES_DE_CATEGORIA.get(valor, valor)


def normalizar_texto(texto: str, nome_ia: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = texto.replace("\u00a0", " ")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r" *\n *", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()

    if nome_ia:
        texto = re.sub(
            r"\bqle\b",
            nome_ia,
            texto,
            flags=re.IGNORECASE,
        )

    return texto


def chave_normalizada(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)

    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )

    texto = texto.lower()
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def hash_conteudo(pergunta: str, resposta: str) -> str:
    base = (
        f"{chave_normalizada(pergunta)}\n"
        f"{chave_normalizada(resposta)}"
    )

    return hashlib.sha256(
        base.encode("utf-8")
    ).hexdigest()


def carregar_categorias_permitidas(
    valor: str | None,
) -> set[str] | None:
    if not valor:
        return None

    categorias = {
        normalizar_categoria(item)
        for item in valor.split(",")
        if item.strip()
    }

    if not categorias:
        return None

    invalidas = sorted(
        categoria
        for categoria in categorias
        if not PADRAO_CATEGORIA.fullmatch(categoria)
    )

    if invalidas:
        raise ErroCorpus(
            "Categorias inválidas em --categorias-permitidas: "
            + ", ".join(invalidas)
        )

    return categorias


def remover_invólucros_acidentais(texto: str) -> str:
    linhas = texto.splitlines()

    while linhas and not linhas[0].strip():
        linhas.pop(0)

    while linhas and not linhas[-1].strip():
        linhas.pop()

    if not linhas:
        return ""

    primeiro = linhas[0].strip().lower()
    ultimo = linhas[-1].strip().lower()

    marcadores_inicio = {
        "```jsonl",
        "```json",
        "```",
        "@'",
        '@"',
    }
    marcadores_fim = {
        "```",
        "'@",
        '"@',
    }

    if primeiro in marcadores_inicio:
        linhas.pop(0)

    if linhas and ultimo in marcadores_fim:
        linhas.pop()

    return "\n".join(linhas).strip()


def ler_entrada(
    caminho: Path,
) -> tuple[list[dict[str, Any]], str]:
    if not caminho.exists():
        raise ErroCorpus(
            f"Arquivo de entrada não encontrado: {caminho.resolve()}"
        )

    if caminho.stat().st_size == 0:
        raise ErroCorpus(
            f"O arquivo de entrada está vazio: {caminho.resolve()}"
        )

    try:
        texto = caminho.read_text(
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError as erro:
        raise ErroCorpus(
            "O dataset não está em UTF-8. "
            "Salve-o como UTF-8 antes de continuar."
        ) from erro

    texto = remover_invólucros_acidentais(texto)

    if not texto:
        raise ErroCorpus(
            "O dataset não contém registros após a limpeza inicial."
        )

    primeiro_caractere = texto.lstrip()[:1]

    if primeiro_caractere == "[":
        try:
            dados = json.loads(texto)
        except json.JSONDecodeError as erro:
            raise ErroCorpus(
                "O arquivo parece ser uma lista JSON, mas está inválido. "
                f"Linha {erro.lineno}, coluna {erro.colno}: {erro.msg}"
            ) from erro

        if not isinstance(dados, list):
            raise ErroCorpus(
                "O JSON principal precisa ser uma lista de objetos."
            )

        registros: list[dict[str, Any]] = []

        for indice, item in enumerate(dados, start=1):
            if not isinstance(item, dict):
                raise ErroCorpus(
                    f"Item {indice} da lista não é um objeto JSON."
                )

            item = dict(item)
            item["_linha_origem"] = indice
            registros.append(item)

        return registros, "json"

    registros = []

    for numero_linha, linha in enumerate(
        texto.splitlines(),
        start=1,
    ):
        if not linha.strip():
            continue

        conteudo = linha.strip()

        if conteudo.endswith(","):
            conteudo = conteudo[:-1].rstrip()

        try:
            item = json.loads(conteudo)
        except json.JSONDecodeError as erro:
            trecho = conteudo[:180]

            raise ErroCorpus(
                "JSONL inválido em "
                f"{caminho}, linha {numero_linha}, "
                f"coluna {erro.colno}: {erro.msg}. "
                f"Trecho: {trecho!r}"
            ) from erro

        if not isinstance(item, dict):
            raise ErroCorpus(
                f"Linha {numero_linha}: era esperado um objeto JSON."
            )

        item = dict(item)
        item["_linha_origem"] = numero_linha
        registros.append(item)

    if not registros:
        raise ErroCorpus(
            "Nenhum objeto JSON foi encontrado no dataset."
        )

    return registros, "jsonl"


def validar_e_limpar(
    itens: list[dict[str, Any]],
    *,
    nome_ia: str,
    categorias_permitidas: set[str] | None,
    minimo_pergunta: int,
    minimo_resposta: int,
    maximo_caracteres: int,
) -> tuple[
    list[Registro],
    list[Problema],
    list[str],
]:
    validos: list[Registro] = []
    problemas: list[Problema] = []
    avisos: list[str] = []

    ids_encontrados: dict[str, int] = {}
    hashes_encontrados: dict[str, int] = {}
    perguntas_para_respostas: dict[str, set[str]] = defaultdict(set)

    for item in itens:
        linha = int(item.get("_linha_origem", 0))
        identificador_bruto = item.get("id")

        identificador = (
            str(identificador_bruto).strip()
            if identificador_bruto is not None
            else ""
        )

        campos_ausentes = sorted(
            campo
            for campo in CAMPOS_OBRIGATORIOS
            if campo not in item
        )

        if campos_ausentes:
            problemas.append(
                Problema(
                    linha=linha,
                    codigo="campos_ausentes",
                    mensagem=(
                        "Campos obrigatórios ausentes: "
                        + ", ".join(campos_ausentes)
                    ),
                    id=identificador or None,
                )
            )
            continue

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
            problemas.append(
                Problema(
                    linha=linha,
                    codigo="id_vazio",
                    mensagem="O campo 'id' está vazio.",
                )
            )
            continue

        if identificador in ids_encontrados:
            problemas.append(
                Problema(
                    linha=linha,
                    codigo="id_duplicado",
                    mensagem=(
                        f"ID duplicado. Primeira ocorrência na linha "
                        f"{ids_encontrados[identificador]}."
                    ),
                    id=identificador,
                )
            )
            continue

        if not categoria:
            problemas.append(
                Problema(
                    linha=linha,
                    codigo="categoria_vazia",
                    mensagem="A categoria está vazia.",
                    id=identificador,
                )
            )
            continue

        if not PADRAO_CATEGORIA.fullmatch(categoria):
            problemas.append(
                Problema(
                    linha=linha,
                    codigo="categoria_invalida",
                    mensagem=(
                        "A categoria normalizada deve conter somente "
                        "letras minúsculas, números e sublinhado, "
                        "com no máximo 64 caracteres."
                    ),
                    id=identificador,
                )
            )
            continue

        if (
            categorias_permitidas is not None
            and categoria not in categorias_permitidas
        ):
            problemas.append(
                Problema(
                    linha=linha,
                    codigo="categoria_nao_permitida",
                    mensagem=(
                        f"Categoria '{categoria}' não está na lista "
                        "fornecida em --categorias-permitidas."
                    ),
                    id=identificador,
                )
            )
            continue

        if len(pergunta) < minimo_pergunta:
            problemas.append(
                Problema(
                    linha=linha,
                    codigo="pergunta_curta",
                    mensagem=(
                        f"A pergunta possui menos de "
                        f"{minimo_pergunta} caracteres."
                    ),
                    id=identificador,
                )
            )
            continue

        if len(resposta) < minimo_resposta:
            problemas.append(
                Problema(
                    linha=linha,
                    codigo="resposta_curta",
                    mensagem=(
                        f"A resposta possui menos de "
                        f"{minimo_resposta} caracteres."
                    ),
                    id=identificador,
                )
            )
            continue

        if (
            len(pergunta) > maximo_caracteres
            or len(resposta) > maximo_caracteres
        ):
            problemas.append(
                Problema(
                    linha=linha,
                    codigo="texto_longo_demais",
                    mensagem=(
                        f"Pergunta ou resposta excede "
                        f"{maximo_caracteres} caracteres."
                    ),
                    id=identificador,
                )
            )
            continue

        marcadores = sorted(
            marcador
            for marcador in MARCADORES_DE_CODIFICACAO_CORROMPIDA
            if marcador in pergunta or marcador in resposta
        )

        if marcadores:
            problemas.append(
                Problema(
                    linha=linha,
                    codigo="codificacao_corrompida",
                    mensagem=(
                        "Foram encontrados sinais de texto corrompido: "
                        + ", ".join(repr(item) for item in marcadores)
                    ),
                    id=identificador,
                )
            )
            continue

        assinatura = hash_conteudo(
            pergunta,
            resposta,
        )

        if assinatura in hashes_encontrados:
            problemas.append(
                Problema(
                    linha=linha,
                    codigo="conteudo_duplicado",
                    mensagem=(
                        "Pergunta e resposta duplicadas. "
                        f"Primeira ocorrência na linha "
                        f"{hashes_encontrados[assinatura]}."
                    ),
                    id=identificador,
                )
            )
            continue

        chave_pergunta = chave_normalizada(pergunta)
        chave_resposta = chave_normalizada(resposta)

        perguntas_para_respostas[
            chave_pergunta
        ].add(chave_resposta)

        ids_encontrados[identificador] = linha
        hashes_encontrados[assinatura] = linha

        validos.append(
            Registro(
                id=identificador,
                categoria=categoria,
                pergunta=pergunta,
                resposta=resposta,
                linha_origem=linha,
            )
        )

    conflitos = sum(
        1
        for respostas in perguntas_para_respostas.values()
        if len(respostas) > 1
    )

    if conflitos:
        avisos.append(
            f"{conflitos} pergunta(s) possuem respostas diferentes. "
            "Essas variantes serão mantidas no mesmo split para evitar "
            "vazamento entre treino, validação e teste."
        )

    return validos, problemas, avisos


def calcular_quantidades_split(
    quantidade_grupos: int,
    fracao_validacao: float,
    fracao_teste: float,
) -> tuple[int, int, int]:
    if quantidade_grupos <= 0:
        return 0, 0, 0

    if quantidade_grupos < 3:
        return quantidade_grupos, 0, 0

    quantidade_validacao = max(
        1,
        round(quantidade_grupos * fracao_validacao),
    )
    quantidade_teste = max(
        1,
        round(quantidade_grupos * fracao_teste),
    )

    while (
        quantidade_validacao
        + quantidade_teste
        >= quantidade_grupos
    ):
        if quantidade_teste >= quantidade_validacao:
            quantidade_teste -= 1
        else:
            quantidade_validacao -= 1

    quantidade_treino = (
        quantidade_grupos
        - quantidade_validacao
        - quantidade_teste
    )

    return (
        quantidade_treino,
        quantidade_validacao,
        quantidade_teste,
    )


def dividir_estratificado(
    registros: list[Registro],
    *,
    seed: int,
    fracao_validacao: float,
    fracao_teste: float,
) -> tuple[
    list[Registro],
    list[Registro],
    list[Registro],
]:
    grupos_por_categoria: dict[
        str,
        dict[str, list[Registro]],
    ] = defaultdict(lambda: defaultdict(list))

    for registro in registros:
        grupos_por_categoria[
            registro.categoria
        ][chave_normalizada(registro.pergunta)].append(registro)

    gerador = random.Random(seed)

    treino: list[Registro] = []
    validacao: list[Registro] = []
    teste: list[Registro] = []

    for categoria in sorted(grupos_por_categoria):
        grupos = [
            grupo
            for _, grupo in sorted(
                grupos_por_categoria[categoria].items(),
                key=lambda item: item[0],
            )
        ]

        gerador.shuffle(grupos)

        (
            quantidade_treino,
            quantidade_validacao,
            _,
        ) = calcular_quantidades_split(
            len(grupos),
            fracao_validacao,
            fracao_teste,
        )

        fim_treino = quantidade_treino
        fim_validacao = (
            quantidade_treino
            + quantidade_validacao
        )

        for grupo in grupos[:fim_treino]:
            treino.extend(grupo)

        for grupo in grupos[
            fim_treino:fim_validacao
        ]:
            validacao.extend(grupo)

        for grupo in grupos[fim_validacao:]:
            teste.extend(grupo)

    gerador.shuffle(treino)
    gerador.shuffle(validacao)
    gerador.shuffle(teste)

    return treino, validacao, teste


def garantir_splits_sem_vazamento(
    treino: list[Registro],
    validacao: list[Registro],
    teste: list[Registro],
) -> None:
    perguntas = {
        "train": {
            chave_normalizada(item.pergunta)
            for item in treino
        },
        "validation": {
            chave_normalizada(item.pergunta)
            for item in validacao
        },
        "test": {
            chave_normalizada(item.pergunta)
            for item in teste
        },
    }

    pares = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )

    for primeiro, segundo in pares:
        intersecao = (
            perguntas[primeiro]
            & perguntas[segundo]
        )

        if intersecao:
            raise ErroCorpus(
                "Foi detectado vazamento de perguntas entre "
                f"{primeiro} e {segundo}: "
                f"{len(intersecao)} ocorrência(s)."
            )


def escrever_texto_atomico(
    caminho: Path,
    conteudo: str,
) -> None:
    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporario = caminho.with_name(
        caminho.name + ".tmp"
    )

    temporario.write_text(
        conteudo,
        encoding="utf-8",
        newline="\n",
    )

    os.replace(
        temporario,
        caminho,
    )


def escrever_json_atomico(
    caminho: Path,
    dados: Any,
) -> None:
    escrever_texto_atomico(
        caminho,
        json.dumps(
            dados,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


def escrever_jsonl_atomico(
    caminho: Path,
    registros: Iterable[Registro],
) -> None:
    linhas = [
        json.dumps(
            registro.para_saida(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for registro in registros
    ]

    conteudo = (
        "\n".join(linhas) + "\n"
        if linhas
        else ""
    )

    escrever_texto_atomico(
        caminho,
        conteudo,
    )


def contar_categorias(
    registros: Iterable[Registro],
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                registro.categoria
                for registro in registros
            ).items()
        )
    )


def calcular_percentuais(
    contagens: dict[str, int],
) -> dict[str, float]:
    total = sum(contagens.values())

    if total == 0:
        return {
            categoria: 0.0
            for categoria in contagens
        }

    return {
        categoria: round(
            quantidade / total * 100,
            4,
        )
        for categoria, quantidade in contagens.items()
    }


def carregar_metas(
    caminho: Path | None,
) -> dict[str, float] | None:
    if caminho is None:
        return None

    if not caminho.exists():
        raise ErroCorpus(
            f"Arquivo de metas não encontrado: {caminho.resolve()}"
        )

    try:
        dados = json.loads(
            caminho.read_text(
                encoding="utf-8-sig"
            )
        )
    except json.JSONDecodeError as erro:
        raise ErroCorpus(
            f"JSON inválido no arquivo de metas: {erro}"
        ) from erro

    if not isinstance(dados, dict):
        raise ErroCorpus(
            "O arquivo de metas precisa ser um objeto JSON."
        )

    metas: dict[str, float] = {}

    for categoria, proporcao in dados.items():
        categoria_normalizada = normalizar_categoria(
            str(categoria)
        )

        try:
            proporcao_numerica = float(proporcao)
        except (TypeError, ValueError) as erro:
            raise ErroCorpus(
                f"Meta inválida para '{categoria}'."
            ) from erro

        if proporcao_numerica < 0:
            raise ErroCorpus(
                f"A meta de '{categoria}' não pode ser negativa."
            )

        metas[categoria_normalizada] = proporcao_numerica

    soma = sum(metas.values())

    if soma <= 0:
        raise ErroCorpus(
            "A soma das metas precisa ser maior que zero."
        )

    return {
        categoria: valor / soma
        for categoria, valor in metas.items()
    }


def comparar_com_metas(
    contagens: dict[str, int],
    metas: dict[str, float] | None,
) -> dict[str, Any] | None:
    if metas is None:
        return None

    total = sum(contagens.values())

    comparacao: dict[str, Any] = {}

    categorias = sorted(
        set(contagens) | set(metas)
    )

    for categoria in categorias:
        real = (
            contagens.get(categoria, 0) / total
            if total
            else 0.0
        )
        meta = metas.get(categoria, 0.0)

        comparacao[categoria] = {
            "quantidade": contagens.get(
                categoria,
                0,
            ),
            "proporcao_real": round(real, 6),
            "proporcao_meta": round(meta, 6),
            "desvio": round(real - meta, 6),
        }

    return comparacao


def escrever_resumo_csv(
    caminho: Path,
    categorias_total: dict[str, int],
    categorias_treino: dict[str, int],
    categorias_validacao: dict[str, int],
    categorias_teste: dict[str, int],
) -> None:
    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporario = caminho.with_name(
        caminho.name + ".tmp"
    )

    categorias = sorted(
        set(categorias_total)
        | set(categorias_treino)
        | set(categorias_validacao)
        | set(categorias_teste)
    )

    with temporario.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(
            (
                "categoria",
                "total",
                "train",
                "validation",
                "test",
            )
        )

        for categoria in categorias:
            escritor.writerow(
                (
                    categoria,
                    categorias_total.get(
                        categoria,
                        0,
                    ),
                    categorias_treino.get(
                        categoria,
                        0,
                    ),
                    categorias_validacao.get(
                        categoria,
                        0,
                    ),
                    categorias_teste.get(
                        categoria,
                        0,
                    ),
                )
            )

    os.replace(
        temporario,
        caminho,
    )


def validar_fracoes(
    validacao: float,
    teste: float,
) -> None:
    if not 0 <= validacao < 1:
        raise ErroCorpus(
            "--fracao-validacao precisa estar entre 0 e 1."
        )

    if not 0 <= teste < 1:
        raise ErroCorpus(
            "--fracao-teste precisa estar entre 0 e 1."
        )

    if validacao + teste >= 0.5:
        raise ErroCorpus(
            "A soma das frações de validação e teste "
            "precisa ser menor que 0.5."
        )


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Valida, normaliza, deduplica e divide o corpus da QLE "
            "em treino, validação e teste."
        )
    )

    parser.add_argument(
        "--entrada",
        type=Path,
        default=Path(
            "data/raw/qle_dataset.jsonl"
        ),
        help=(
            "Dataset de entrada em JSONL ou lista JSON. "
            "Padrão: data/raw/qle_dataset.jsonl"
        ),
    )

    parser.add_argument(
        "--saida-processado",
        type=Path,
        default=Path(
            "data/processed/qle_limpo.jsonl"
        ),
    )

    parser.add_argument(
        "--saida-relatorio",
        type=Path,
        default=Path(
            "data/processed/relatorio_corpus.json"
        ),
    )

    parser.add_argument(
        "--saida-resumo-csv",
        type=Path,
        default=Path(
            "data/processed/resumo_categorias.csv"
        ),
    )

    parser.add_argument(
        "--saida-treino",
        type=Path,
        default=Path(
            "data/splits/train.jsonl"
        ),
    )

    parser.add_argument(
        "--saida-validacao",
        type=Path,
        default=Path(
            "data/splits/validation.jsonl"
        ),
    )

    parser.add_argument(
        "--saida-teste",
        type=Path,
        default=Path(
            "data/splits/test.jsonl"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--nome-ia",
        type=str,
        default="QLE",
    )

    parser.add_argument(
        "--fracao-validacao",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--fracao-teste",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--categorias-permitidas",
        type=str,
        default=None,
        help=(
            "Lista opcional separada por vírgulas. "
            "Se omitida, qualquer categoria normalizada é aceita."
        ),
    )

    parser.add_argument(
        "--metas",
        type=Path,
        default=None,
        help=(
            "Arquivo JSON opcional com proporções desejadas "
            "por categoria."
        ),
    )

    parser.add_argument(
        "--minimo-pergunta",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--minimo-resposta",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--maximo-caracteres",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--estrito",
        action="store_true",
        help=(
            "Interrompe sem substituir os splits quando existir "
            "qualquer registro inválido."
        ),
    )

    parser.add_argument(
        "--mostrar-erros",
        type=int,
        default=50,
        help=(
            "Quantidade máxima de problemas mostrados no terminal. "
            "Todos permanecem no relatório JSON."
        ),
    )

    return parser


def executar(args: argparse.Namespace) -> int:
    validar_fracoes(
        args.fracao_validacao,
        args.fracao_teste,
    )

    if args.minimo_pergunta < 1:
        raise ErroCorpus(
            "--minimo-pergunta precisa ser pelo menos 1."
        )

    if args.minimo_resposta < 1:
        raise ErroCorpus(
            "--minimo-resposta precisa ser pelo menos 1."
        )

    if args.maximo_caracteres < 10:
        raise ErroCorpus(
            "--maximo-caracteres precisa ser pelo menos 10."
        )

    categorias_permitidas = (
        carregar_categorias_permitidas(
            args.categorias_permitidas
        )
    )
    metas = carregar_metas(args.metas)

    itens_brutos, formato_entrada = ler_entrada(
        args.entrada
    )

    validos, problemas, avisos = validar_e_limpar(
        itens_brutos,
        nome_ia=args.nome_ia,
        categorias_permitidas=categorias_permitidas,
        minimo_pergunta=args.minimo_pergunta,
        minimo_resposta=args.minimo_resposta,
        maximo_caracteres=args.maximo_caracteres,
    )

    categorias_validas = contar_categorias(
        validos
    )

    relatorio_base: dict[str, Any] = {
        "status": (
            "falhou"
            if args.estrito and problemas
            else "concluido"
        ),
        "gerado_em_utc": agora_utc_iso(),
        "entrada": {
            "caminho": str(
                args.entrada.resolve()
            ),
            "formato": formato_entrada,
            "tamanho_bytes": (
                args.entrada.stat().st_size
            ),
            "sha256": sha256_arquivo(
                args.entrada
            ),
        },
        "configuracao": {
            "seed": args.seed,
            "nome_ia": args.nome_ia,
            "fracao_validacao": (
                args.fracao_validacao
            ),
            "fracao_teste": args.fracao_teste,
            "modo_estrito": args.estrito,
            "categorias_permitidas": (
                sorted(categorias_permitidas)
                if categorias_permitidas
                else None
            ),
        },
        "contagens": {
            "brutos": len(itens_brutos),
            "validos": len(validos),
            "invalidos_ou_duplicados": (
                len(problemas)
            ),
        },
        "categorias": {
            "total": categorias_validas,
            "percentual_total": (
                calcular_percentuais(
                    categorias_validas
                )
            ),
        },
        "comparacao_com_metas": (
            comparar_com_metas(
                categorias_validas,
                metas,
            )
        ),
        "avisos": avisos,
        "problemas": [
            asdict(problema)
            for problema in problemas
        ],
    }

    if args.estrito and problemas:
        escrever_json_atomico(
            args.saida_relatorio,
            relatorio_base,
        )

        print("=" * 72)
        print("CORPUS QLE REPROVADO NO MODO ESTRITO")
        print("=" * 72)
        print(
            f"Registros brutos: {len(itens_brutos):,}"
        )
        print(
            f"Registros válidos: {len(validos):,}"
        )
        print(
            f"Problemas: {len(problemas):,}"
        )
        print(
            "Os arquivos de treino, validação e teste "
            "não foram substituídos."
        )
        print(
            f"Relatório: "
            f"{args.saida_relatorio.resolve()}"
        )

        limite = max(
            0,
            args.mostrar_erros,
        )

        for problema in problemas[:limite]:
            identificador = (
                f" | id={problema.id}"
                if problema.id
                else ""
            )
            print(
                f"- linha {problema.linha} "
                f"[{problema.codigo}] "
                f"{problema.mensagem}"
                f"{identificador}"
            )

        return 2

    if not validos:
        raise ErroCorpus(
            "Nenhum registro válido restou após a validação."
        )

    treino, validacao, teste = dividir_estratificado(
        validos,
        seed=args.seed,
        fracao_validacao=args.fracao_validacao,
        fracao_teste=args.fracao_teste,
    )

    if not treino:
        raise ErroCorpus(
            "O split de treino ficou vazio."
        )

    garantir_splits_sem_vazamento(
        treino,
        validacao,
        teste,
    )

    escrever_jsonl_atomico(
        args.saida_processado,
        validos,
    )
    escrever_jsonl_atomico(
        args.saida_treino,
        treino,
    )
    escrever_jsonl_atomico(
        args.saida_validacao,
        validacao,
    )
    escrever_jsonl_atomico(
        args.saida_teste,
        teste,
    )

    categorias_treino = contar_categorias(
        treino
    )
    categorias_validacao = contar_categorias(
        validacao
    )
    categorias_teste = contar_categorias(
        teste
    )

    relatorio_base["splits"] = {
        "train": {
            "quantidade": len(treino),
            "categorias": categorias_treino,
            "sha256": sha256_arquivo(
                args.saida_treino
            ),
        },
        "validation": {
            "quantidade": len(validacao),
            "categorias": categorias_validacao,
            "sha256": sha256_arquivo(
                args.saida_validacao
            ),
        },
        "test": {
            "quantidade": len(teste),
            "categorias": categorias_teste,
            "sha256": sha256_arquivo(
                args.saida_teste
            ),
        },
    }

    relatorio_base["saidas"] = {
        "processado": str(
            args.saida_processado.resolve()
        ),
        "treino": str(
            args.saida_treino.resolve()
        ),
        "validacao": str(
            args.saida_validacao.resolve()
        ),
        "teste": str(
            args.saida_teste.resolve()
        ),
        "resumo_csv": str(
            args.saida_resumo_csv.resolve()
        ),
    }

    escrever_resumo_csv(
        args.saida_resumo_csv,
        categorias_validas,
        categorias_treino,
        categorias_validacao,
        categorias_teste,
    )

    escrever_json_atomico(
        args.saida_relatorio,
        relatorio_base,
    )

    print("=" * 72)
    print("CORPUS QLE PREPARADO COM SUCESSO")
    print("=" * 72)
    print(
        f"Entrada: {args.entrada.resolve()}"
    )
    print(
        f"Formato detectado: {formato_entrada}"
    )
    print(
        f"Registros brutos: {len(itens_brutos):,}"
    )
    print(
        f"Registros válidos: {len(validos):,}"
    )
    print(
        f"Removidos/ignorados: {len(problemas):,}"
    )
    print(
        f"Treino: {len(treino):,}"
    )
    print(
        f"Validação: {len(validacao):,}"
    )
    print(
        f"Teste: {len(teste):,}"
    )
    print("\nCategorias:")

    for categoria, quantidade in categorias_validas.items():
        percentual = (
            quantidade / len(validos) * 100
        )
        print(
            f"- {categoria:<28} "
            f"{quantidade:>6} "
            f"({percentual:>6.2f}%)"
        )

    if avisos:
        print("\nAvisos:")

        for aviso in avisos:
            print(f"- {aviso}")

    if problemas and not args.estrito:
        print(
            f"\n{len(problemas)} registro(s) inválido(s) "
            "foram ignorados."
        )

    print(
        f"\nRelatório JSON: "
        f"{args.saida_relatorio.resolve()}"
    )
    print(
        f"Resumo CSV: "
        f"{args.saida_resumo_csv.resolve()}"
    )

    return 0


def main() -> None:
    parser = construir_parser()
    args = parser.parse_args()

    try:
        codigo_saida = executar(args)
    except ErroCorpus as erro:
        print(
            f"ERRO: {erro}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print(
            "\nOperação cancelada pelo usuário.",
            file=sys.stderr,
        )
        raise SystemExit(130) from None

    raise SystemExit(codigo_saida)


if __name__ == "__main__":
    main()
