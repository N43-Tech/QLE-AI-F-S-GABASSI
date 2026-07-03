from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch

from src.model import ConfigModelo, MiniLLM
from src.tokenizer_bpe import TokenizadorBPE


FORMATO_CHECKPOINT = "qle_bpe_v1"
NOME_IA = "QLE"


class ErroGeracao(RuntimeError):
    """Erro controlado de carregamento ou geração."""


def configurar_utf8() -> None:
    for fluxo in (
        sys.stdout,
        sys.stderr,
        sys.stdin,
    ):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(
                encoding="utf-8"
            )


def escolher_dispositivo(
    solicitado: str = "auto",
) -> torch.device:
    solicitado = solicitado.lower()

    if solicitado == "cpu":
        return torch.device("cpu")

    if solicitado == "cuda":
        if not torch.cuda.is_available():
            raise ErroGeracao(
                "CUDA foi solicitada, mas não está disponível."
            )
        return torch.device("cuda")

    if solicitado == "mps":
        disponivel = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

        if not disponivel:
            raise ErroGeracao(
                "MPS foi solicitado, mas não está disponível."
            )
        return torch.device("mps")

    if solicitado != "auto":
        raise ErroGeracao(
            f"Dispositivo inválido: {solicitado}."
        )

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def sha256_arquivo(
    caminho: Path,
) -> str:
    digest = hashlib.sha256()

    with caminho.open("rb") as arquivo:
        for bloco in iter(
            lambda: arquivo.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(bloco)

    return digest.hexdigest()


def carregar_checkpoint(
    caminho: Path,
    dispositivo: torch.device,
) -> dict[str, Any]:
    caminho = caminho.resolve()

    if not caminho.exists():
        raise ErroGeracao(
            f"Checkpoint não encontrado: {caminho}"
        )

    if caminho.stat().st_size == 0:
        raise ErroGeracao(
            f"O checkpoint está vazio: {caminho}"
        )

    try:
        checkpoint = torch.load(
            caminho,
            map_location=dispositivo,
            weights_only=False,
        )
    except (
        EOFError,
        OSError,
        RuntimeError,
    ) as erro:
        raise ErroGeracao(
            "Não foi possível carregar o checkpoint. "
            f"Tamanho: {caminho.stat().st_size} bytes. "
            "O arquivo pode estar corrompido."
        ) from erro

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ErroGeracao(
            "O checkpoint não possui formato de dicionário."
        )

    formato = checkpoint.get(
        "formato_checkpoint"
    )

    if formato != FORMATO_CHECKPOINT:
        tokenizador_antigo = checkpoint.get(
            "tokenizador"
        )

        raise ErroGeracao(
            "Checkpoint incompatível com o pipeline BPE. "
            f"Formato encontrado: {formato!r}; "
            f"tokenizador encontrado: {tokenizador_antigo!r}. "
            "Use um checkpoint criado pelo novo src.train."
        )

    for chave in (
        "config_modelo",
        "estado_modelo",
    ):
        if chave not in checkpoint:
            raise ErroGeracao(
                f"O checkpoint não possui a chave '{chave}'."
            )

    return checkpoint


def resolver_caminho_tokenizador(
    *,
    informado: Path | None,
    checkpoint: dict[str, Any],
    caminho_checkpoint: Path,
) -> Path:
    if informado is not None:
        candidatos = [
            informado,
        ]
    else:
        meta = checkpoint.get(
            "tokenizador",
            {},
        )
        valor_meta = meta.get(
            "caminho"
        )

        candidatos: list[Path] = []

        if valor_meta:
            caminho_meta = Path(
                str(valor_meta)
            )

            candidatos.extend(
                (
                    caminho_meta,
                    Path.cwd() / caminho_meta,
                    caminho_checkpoint
                    .resolve()
                    .parent
                    .parent
                    / caminho_meta,
                )
            )

        candidatos.append(
            Path(
                "tokenizer/qle_bpe_2000.json"
            )
        )

    vistos: set[str] = set()

    for candidato in candidatos:
        resolvido = candidato.expanduser()

        chave = str(
            resolvido.resolve(
                strict=False
            )
        )

        if chave in vistos:
            continue

        vistos.add(chave)

        if (
            resolvido.exists()
            and resolvido.is_file()
        ):
            return resolvido

    caminhos = "\n".join(
        f"- {item}"
        for item in candidatos
    )

    raise ErroGeracao(
        "Tokenizador BPE não encontrado. "
        "Caminhos verificados:\n"
        + caminhos
    )


def carregar_modelo_e_tokenizador(
    *,
    caminho_checkpoint: Path,
    caminho_tokenizador: Path | None,
    dispositivo: torch.device,
) -> tuple[
    MiniLLM,
    ConfigModelo,
    TokenizadorBPE,
    dict[str, Any],
    Path,
]:
    checkpoint = carregar_checkpoint(
        caminho_checkpoint,
        dispositivo,
    )

    tokenizador_path = (
        resolver_caminho_tokenizador(
            informado=caminho_tokenizador,
            checkpoint=checkpoint,
            caminho_checkpoint=(
                caminho_checkpoint
            ),
        )
    )

    tokenizador = TokenizadorBPE(
        tokenizador_path
    )

    cfg = ConfigModelo(
        **checkpoint["config_modelo"]
    )
    cfg.validar()

    if (
        tokenizador.tamanho_vocab
        != cfg.tamanho_vocab
    ):
        raise ErroGeracao(
            "O vocabulário do tokenizador "
            f"({tokenizador.tamanho_vocab}) difere do modelo "
            f"({cfg.tamanho_vocab})."
        )

    meta_tokenizador = checkpoint.get(
        "tokenizador",
        {},
    )
    hash_salvo = meta_tokenizador.get(
        "sha256"
    )

    if (
        hash_salvo
        and hash_salvo
        != sha256_arquivo(
            tokenizador_path
        )
    ):
        raise ErroGeracao(
            "O tokenizador informado não é o mesmo utilizado "
            "no treinamento do checkpoint."
        )

    modelo = MiniLLM(
        cfg
    ).to(dispositivo)

    modelo.load_state_dict(
        checkpoint["estado_modelo"],
        strict=True,
    )
    modelo.eval()

    return (
        modelo,
        cfg,
        tokenizador,
        checkpoint,
        tokenizador_path,
    )


def construir_prompt_ids(
    *,
    mensagem: str,
    tokenizador: TokenizadorBPE,
    cfg: ConfigModelo,
) -> list[int]:
    mensagem = mensagem.strip()

    if not mensagem:
        raise ErroGeracao(
            "A mensagem está vazia."
        )

    pergunta_ids = tokenizador.codificar(
        mensagem
    )

    # <BOS>, <USER> e <ASSISTANT>.
    limite_pergunta = max(
        1,
        cfg.comprimento_max - 3,
    )

    if len(pergunta_ids) > limite_pergunta:
        pergunta_ids = pergunta_ids[
            -limite_pergunta:
        ]

    return [
        tokenizador.especiais.bos,
        tokenizador.especiais.usuario,
        *pergunta_ids,
        tokenizador.especiais.assistente,
    ]


def aplicar_penalidade_repeticao(
    logits: torch.Tensor,
    tokens_gerados: list[int],
    penalidade: float,
) -> None:
    if penalidade == 1.0:
        return

    for token_id in set(
        tokens_gerados
    ):
        valor = logits[token_id]

        if valor < 0:
            logits[token_id] = (
                valor * penalidade
            )
        else:
            logits[token_id] = (
                valor / penalidade
            )


def tokens_bloqueados_por_ngram(
    tokens: list[int],
    tamanho_ngram: int,
) -> set[int]:
    if tamanho_ngram <= 0:
        return set()

    if tamanho_ngram == 1:
        return set(tokens)

    tamanho_prefixo = (
        tamanho_ngram - 1
    )

    if len(tokens) < tamanho_prefixo:
        return set()

    prefixo_atual = tuple(
        tokens[-tamanho_prefixo:]
    )

    bloqueados: set[int] = set()

    limite = (
        len(tokens)
        - tamanho_ngram
        + 1
    )

    for inicio in range(
        max(0, limite)
    ):
        prefixo = tuple(
            tokens[
                inicio:
                inicio + tamanho_prefixo
            ]
        )

        if prefixo == prefixo_atual:
            proximo_indice = (
                inicio + tamanho_prefixo
            )

            if proximo_indice < len(tokens):
                bloqueados.add(
                    tokens[
                        proximo_indice
                    ]
                )

    return bloqueados


def filtrar_top_k(
    logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    if top_k <= 0:
        return logits

    k = min(
        top_k,
        logits.numel(),
    )

    limite = torch.topk(
        logits,
        k,
    ).values[-1]

    return logits.masked_fill(
        logits < limite,
        float("-inf"),
    )


def filtrar_top_p(
    logits: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    if top_p >= 1.0:
        return logits

    logits_ordenados, indices = torch.sort(
        logits,
        descending=True,
    )

    probabilidades = torch.softmax(
        logits_ordenados,
        dim=-1,
    )
    acumuladas = torch.cumsum(
        probabilidades,
        dim=-1,
    )

    remover = acumuladas > top_p
    remover[1:] = remover[:-1].clone()
    remover[0] = False

    logits_ordenados = (
        logits_ordenados.masked_fill(
            remover,
            float("-inf"),
        )
    )

    resultado = torch.full_like(
        logits,
        float("-inf"),
    )

    resultado.scatter_(
        0,
        indices,
        logits_ordenados,
    )

    return resultado


def amostrar_token(
    *,
    logits: torch.Tensor,
    temperatura: float,
    top_k: int,
    top_p: float,
) -> int:
    if temperatura == 0:
        return int(
            torch.argmax(
                logits
            ).item()
        )

    logits = logits / temperatura
    logits = filtrar_top_k(
        logits,
        top_k,
    )
    logits = filtrar_top_p(
        logits,
        top_p,
    )

    if not torch.isfinite(
        logits
    ).any():
        raise ErroGeracao(
            "Todos os tokens foram bloqueados pelos filtros de geração."
        )

    probabilidades = torch.softmax(
        logits,
        dim=-1,
    )

    if not torch.isfinite(
        probabilidades
    ).all():
        raise ErroGeracao(
            "Probabilidades não finitas durante a geração."
        )

    return int(
        torch.multinomial(
            probabilidades,
            num_samples=1,
        ).item()
    )


@torch.inference_mode()
def gerar_tokens(
    *,
    modelo: MiniLLM,
    cfg: ConfigModelo,
    tokenizador: TokenizadorBPE,
    prompt_ids: list[int],
    quantidade_tokens: int,
    temperatura: float,
    top_k: int,
    top_p: float,
    penalidade_repeticao: float,
    no_repeat_ngram: int,
) -> tuple[list[int], bool]:
    dispositivo = next(
        modelo.parameters()
    ).device

    sequencia = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=dispositivo,
    )

    gerados: list[int] = []
    atingiu_eos = False

    ids_proibidos = {
        tokenizador.especiais.pad,
        tokenizador.especiais.bos,
        tokenizador.especiais.usuario,
        tokenizador.especiais.assistente,
        tokenizador.especiais.unk,
    }

    for _ in range(
        quantidade_tokens
    ):
        entrada = sequencia[
            :,
            -cfg.comprimento_max:,
        ]

        logits, _ = modelo(
            entrada
        )
        proximos_logits = (
            logits[0, -1, :]
            .float()
            .clone()
        )

        aplicar_penalidade_repeticao(
            proximos_logits,
            gerados,
            penalidade_repeticao,
        )

        bloqueados_ngram = (
            tokens_bloqueados_por_ngram(
                gerados,
                no_repeat_ngram,
            )
        )

        for token_id in (
            ids_proibidos
            | bloqueados_ngram
        ):
            proximos_logits[
                token_id
            ] = float("-inf")

        if not torch.isfinite(
            proximos_logits
        ).any():
            proximo_id = (
                tokenizador
                .especiais
                .eos
            )
        else:
            proximo_id = amostrar_token(
                logits=proximos_logits,
                temperatura=temperatura,
                top_k=top_k,
                top_p=top_p,
            )

        if (
            proximo_id
            == tokenizador.especiais.eos
        ):
            atingiu_eos = True
            break

        gerados.append(
            proximo_id
        )

        novo = torch.tensor(
            [[proximo_id]],
            dtype=torch.long,
            device=dispositivo,
        )

        sequencia = torch.cat(
            (sequencia, novo),
            dim=1,
        )

    return gerados, atingiu_eos


def gerar_resposta(
    *,
    modelo: MiniLLM,
    cfg: ConfigModelo,
    tokenizador: TokenizadorBPE,
    mensagem: str,
    quantidade_tokens: int,
    temperatura: float,
    top_k: int,
    top_p: float,
    penalidade_repeticao: float,
    no_repeat_ngram: int,
) -> tuple[str, bool, int]:
    prompt_ids = construir_prompt_ids(
        mensagem=mensagem,
        tokenizador=tokenizador,
        cfg=cfg,
    )

    gerados, atingiu_eos = gerar_tokens(
        modelo=modelo,
        cfg=cfg,
        tokenizador=tokenizador,
        prompt_ids=prompt_ids,
        quantidade_tokens=(
            quantidade_tokens
        ),
        temperatura=temperatura,
        top_k=top_k,
        top_p=top_p,
        penalidade_repeticao=(
            penalidade_repeticao
        ),
        no_repeat_ngram=(
            no_repeat_ngram
        ),
    )

    texto = tokenizador.decodificar(
        gerados,
        ignorar_especiais=True,
    ).strip()

    if not texto:
        texto = (
            "[A QLE não gerou uma resposta textual.]"
        )

    return (
        texto,
        atingiu_eos,
        len(gerados),
    )


def validar_argumentos(
    args: argparse.Namespace,
) -> None:
    if args.tokens <= 0:
        raise ErroGeracao(
            "--tokens precisa ser positivo."
        )

    if args.temperatura < 0:
        raise ErroGeracao(
            "--temperatura não pode ser negativa."
        )

    if args.top_k < 0:
        raise ErroGeracao(
            "--top-k não pode ser negativo."
        )

    if not 0 < args.top_p <= 1:
        raise ErroGeracao(
            "--top-p precisa estar entre 0 e 1."
        )

    if args.penalidade_repeticao < 1:
        raise ErroGeracao(
            "--penalidade-repeticao precisa ser pelo menos 1."
        )

    if args.no_repeat_ngram < 0:
        raise ErroGeracao(
            "--no-repeat-ngram não pode ser negativo."
        )

    if (
        args.mensagem is not None
        and args.prompt is not None
    ):
        raise ErroGeracao(
            "Use apenas --mensagem ou --prompt."
        )


def criar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Executa a QLE treinada com BPE."
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
        help=(
            "Opcional. Quando omitido, o caminho é lido "
            "dos metadados do checkpoint."
        ),
    )

    parser.add_argument(
        "--mensagem",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Alias técnico de --mensagem. O formato especial "
            "é adicionado automaticamente."
        ),
    )

    parser.add_argument(
        "--chat",
        action="store_true",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--temperatura",
        type=float,
        default=0.4,
        help="Use 0 para geração gulosa.",
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
        "--seed",
        type=int,
        default=42,
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

    parser.add_argument(
        "--mostrar-metadados",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    configurar_utf8()
    args = criar_argumentos()
    validar_argumentos(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

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

    if args.mostrar_metadados:
        print("=" * 64)
        print("QLE — METADADOS")
        print("=" * 64)
        print(
            f"Checkpoint: "
            f"{args.checkpoint.resolve()}"
        )
        print(
            f"Tokenizador: "
            f"{tokenizador_path.resolve()}"
        )
        print(
            f"Dispositivo: {dispositivo}"
        )
        print(
            f"Passo: "
            f"{checkpoint.get('passo_atual', '?')}"
        )
        print(
            f"Parâmetros: "
            f"{modelo.contar_parametros():,}"
        )
        print(
            f"Vocabulário: "
            f"{tokenizador.tamanho_vocab:,}"
        )
        print("=" * 64)

    def responder(
        mensagem: str,
    ) -> None:
        texto, eos, tokens = gerar_resposta(
            modelo=modelo,
            cfg=cfg,
            tokenizador=tokenizador,
            mensagem=mensagem,
            quantidade_tokens=args.tokens,
            temperatura=args.temperatura,
            top_k=args.top_k,
            top_p=args.top_p,
            penalidade_repeticao=(
                args.penalidade_repeticao
            ),
            no_repeat_ngram=(
                args.no_repeat_ngram
            ),
        )

        print(f"{NOME_IA}: {texto}")

        if args.mostrar_metadados:
            print(
                f"[tokens={tokens} | eos={eos}]"
            )

    if args.chat:
        print("=" * 64)
        print("QLE iniciada")
        print(f"Dispositivo: {dispositivo}")
        print("Digite 'sair' para encerrar.")
        print("=" * 64)

        while True:
            try:
                mensagem = input(
                    "\nVocê: "
                ).strip()
            except (
                EOFError,
                KeyboardInterrupt,
            ):
                print(
                    "\nConversa encerrada."
                )
                break

            if mensagem.lower() in {
                "sair",
                "exit",
                "quit",
                "encerrar",
            }:
                print(
                    "Conversa encerrada."
                )
                break

            if not mensagem:
                continue

            responder(mensagem)

        return

    mensagem = (
        args.mensagem
        if args.mensagem is not None
        else args.prompt
    )

    if mensagem is None:
        mensagem = "Olá."

    responder(mensagem)


if __name__ == "__main__":
    try:
        main()
    except ErroGeracao as erro:
        print(
            f"ERRO: {erro}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
