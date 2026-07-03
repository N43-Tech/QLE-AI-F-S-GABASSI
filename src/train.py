from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset import DatasetCausal
from src.model import ConfigModelo, MiniLLM
from src.tokenizer import TokenizadorBytes


TOKENIZADOR_VERSAO = "bytes_utf8_v1"


def escolher_dispositivo() -> torch.device:
    """
    Escolhe automaticamente o melhor dispositivo disponível.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def configurar_sementes(semente: int) -> None:
    """
    Configura sementes para tornar os experimentos reproduzíveis.
    """

    random.seed(semente)
    torch.manual_seed(semente)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


def carregar_config(
    caminho: Path,
) -> tuple[ConfigModelo, dict[str, Any]]:
    """
    Carrega a configuração JSON do modelo e do treinamento.
    """

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {caminho}"
        )

    dados = json.loads(
        caminho.read_text(encoding="utf-8-sig")
    )

    if "modelo" not in dados:
        raise KeyError(
            "A configuração não possui a seção 'modelo'."
        )

    if "treino" not in dados:
        raise KeyError(
            "A configuração não possui a seção 'treino'."
        )

    configuracao_modelo = ConfigModelo(
        **dados["modelo"]
    )

    configuracao_treino = dict(
        dados["treino"]
    )

    return configuracao_modelo, configuracao_treino


def validar_configuracao_treino(
    configuracao: dict[str, Any],
) -> None:
    """
    Verifica os parâmetros obrigatórios do treinamento.
    """

    obrigatorios = (
        "batch_size",
        "passos",
        "warmup_passos",
        "lr_max",
        "lr_min",
        "weight_decay",
        "grad_clip",
        "log_intervalo",
    )

    ausentes = [
        nome
        for nome in obrigatorios
        if nome not in configuracao
    ]

    if ausentes:
        raise KeyError(
            "Parâmetros ausentes na seção 'treino': "
            + ", ".join(ausentes)
        )

    if int(configuracao["batch_size"]) <= 0:
        raise ValueError(
            "batch_size precisa ser maior que zero."
        )

    if int(configuracao["passos"]) <= 0:
        raise ValueError(
            "passos precisa ser maior que zero."
        )

    if float(configuracao["lr_max"]) <= 0:
        raise ValueError(
            "lr_max precisa ser maior que zero."
        )

    if float(configuracao["lr_min"]) < 0:
        raise ValueError(
            "lr_min não pode ser negativo."
        )

    if float(configuracao["grad_clip"]) <= 0:
        raise ValueError(
            "grad_clip precisa ser maior que zero."
        )


def taxa_aprendizado(
    passo: int,
    total: int,
    warmup: int,
    lr_max: float,
    lr_min: float,
) -> float:
    """
    Warmup linear seguido de decaimento cosseno.
    """

    if passo < warmup:
        return (
            lr_max
            * (passo + 1)
            / max(1, warmup)
        )

    progresso = (
        (passo - warmup)
        / max(1, total - warmup)
    )

    progresso = min(
        max(progresso, 0.0),
        1.0,
    )

    fator_cosseno = 0.5 * (
        1.0
        + math.cos(math.pi * progresso)
    )

    return (
        lr_min
        + (lr_max - lr_min)
        * fator_cosseno
    )


def perplexidade_segura(perda: float) -> float:
    """
    Calcula perplexidade evitando overflow numérico.
    """

    return math.exp(
        min(perda, 20.0)
    )


def verificar_texto_corrompido(texto: str) -> None:
    """
    Emite um aviso quando encontra sinais comuns de texto
    UTF-8 interpretado incorretamente.
    """

    sinais = (
        "UsuÃ",
        "vocÃ",
        "inteligÃ",
        "automaÃ",
        "├",
        "�",
    )

    encontrados = [
        sinal
        for sinal in sinais
        if sinal in texto
    ]

    if encontrados:
        print(
            "\nAVISO: o corpus pode conter caracteres "
            "corrompidos."
        )

        print(
            "Padrões encontrados:",
            ", ".join(repr(x) for x in encontrados),
        )

        print(
            "Corrija o corpus antes de realizar um "
            "treinamento longo.\n"
        )


def dividir_tokens(
    tokens: list[int],
    comprimento_contexto: int,
    fracao_validacao: float,
) -> tuple[list[int], list[int]]:
    """
    Divide o corpus em treino e validação.

    A validação utiliza o trecho final do corpus, evitando que
    as mesmas janelas apareçam nos dois conjuntos.
    """

    if not 0.0 < fracao_validacao < 0.5:
        raise ValueError(
            "A fração de validação deve estar entre 0 e 0.5."
        )

    minimo = comprimento_contexto + 2

    if len(tokens) < minimo * 2:
        raise ValueError(
            "Corpus pequeno demais para criar treino e validação. "
            f"São necessários pelo menos {minimo * 2} tokens."
        )

    quantidade_validacao = max(
        minimo,
        int(len(tokens) * fracao_validacao),
    )

    quantidade_validacao = min(
        quantidade_validacao,
        len(tokens) - minimo,
    )

    ponto_corte = (
        len(tokens)
        - quantidade_validacao
    )

    tokens_treino = tokens[:ponto_corte]
    tokens_validacao = tokens[ponto_corte:]

    return tokens_treino, tokens_validacao


def criar_loader(
    tokens: list[int],
    comprimento_contexto: int,
    batch_size: int,
    embaralhar: bool,
    dispositivo: torch.device,
    semente: int,
    num_workers: int,
) -> DataLoader:
    """
    Cria um DataLoader para modelagem causal.
    """

    dataset = DatasetCausal(
        tokens,
        comprimento_contexto,
    )

    if len(dataset) == 0:
        raise ValueError(
            "O dataset ficou vazio. Aumente o corpus ou "
            "reduza comprimento_max."
        )

    batch_real = min(
        batch_size,
        len(dataset),
    )

    gerador = torch.Generator()
    gerador.manual_seed(semente)

    return DataLoader(
        dataset,
        batch_size=batch_real,
        shuffle=embaralhar,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=dispositivo.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=gerador,
    )


def proximo_lote(
    iterador: Any,
    loader: DataLoader,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """
    Obtém o próximo lote e reinicia o loader quando necessário.
    """

    try:
        x, y = next(iterador)
    except StopIteration:
        iterador = iter(loader)
        x, y = next(iterador)

    return iterador, x, y


def avaliar_modelo(
    modelo: MiniLLM,
    loader: DataLoader,
    dispositivo: torch.device,
    max_lotes: int,
) -> float:
    """
    Calcula a loss média no conjunto de validação.
    """

    modelo.eval()

    perdas: list[float] = []

    with torch.inference_mode():
        for indice, (x, y) in enumerate(loader):
            if indice >= max_lotes:
                break

            x = x.to(
                dispositivo,
                non_blocking=True,
            )

            y = y.to(
                dispositivo,
                non_blocking=True,
            )

            _, perda = modelo(x, y)

            if perda is None:
                raise RuntimeError(
                    "O modelo não retornou loss na validação."
                )

            perdas.append(
                float(perda.item())
            )

    modelo.train()

    if not perdas:
        raise RuntimeError(
            "Nenhum lote foi avaliado."
        )

    return sum(perdas) / len(perdas)


def mover_estado_otimizador(
    otimizador: AdamW,
    dispositivo: torch.device,
) -> None:
    """
    Move os tensores internos do otimizador para o dispositivo.
    """

    for estado in otimizador.state.values():
        for chave, valor in estado.items():
            if isinstance(valor, torch.Tensor):
                estado[chave] = valor.to(
                    dispositivo
                )


def salvar_checkpoint(
    caminho: Path,
    modelo: MiniLLM,
    otimizador: AdamW,
    configuracao_modelo: ConfigModelo,
    configuracao_treino: dict[str, Any],
    passo_atual: int,
    melhor_val_loss: float,
    tokens_treino: int,
    tokens_validacao: int,
) -> None:
    """
    Salva pesos, otimizador, configuração e progresso.

    O arquivo temporário reduz o risco de deixar um checkpoint
    corrompido caso a gravação seja interrompida.
    """

    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporario = caminho.with_suffix(
        caminho.suffix + ".tmp"
    )

    dados: dict[str, Any] = {
        "config_modelo": configuracao_modelo.para_dict(),
        "config_treino": configuracao_treino,
        "estado_modelo": modelo.state_dict(),
        "estado_otimizador": otimizador.state_dict(),
        "tokenizador": TOKENIZADOR_VERSAO,
        "passo_atual": passo_atual,
        "passos": passo_atual,
        "melhor_val_loss": melhor_val_loss,
        "tokens_treino": tokens_treino,
        "tokens_validacao": tokens_validacao,
        "rng_python": random.getstate(),
        "rng_torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        dados["rng_cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    torch.save(
        dados,
        temporario,
    )

    temporario.replace(caminho)


def restaurar_checkpoint(
    caminho: Path,
    modelo: MiniLLM,
    otimizador: AdamW,
    configuracao_modelo: ConfigModelo,
    dispositivo: torch.device,
) -> tuple[int, float]:
    """
    Restaura pesos e, quando disponível, o estado do otimizador.
    """

    if not caminho.exists():
        raise FileNotFoundError(
            f"Checkpoint para retomada não encontrado: {caminho}"
        )

    print(
        f"Carregando checkpoint: {caminho.resolve()}"
    )

    checkpoint = torch.load(
        caminho,
        map_location=dispositivo,
        weights_only=False,
    )

    if "estado_modelo" not in checkpoint:
        raise KeyError(
            "O checkpoint não possui 'estado_modelo'."
        )

    config_checkpoint = checkpoint.get(
        "config_modelo"
    )

    config_atual = (
        configuracao_modelo.para_dict()
    )

    if (
        config_checkpoint is not None
        and config_checkpoint != config_atual
    ):
        raise ValueError(
            "A arquitetura do checkpoint é diferente da "
            "configuração atual.\n"
            f"Checkpoint: {config_checkpoint}\n"
            f"Configuração atual: {config_atual}"
        )

    modelo.load_state_dict(
        checkpoint["estado_modelo"]
    )

    if "estado_otimizador" in checkpoint:
        otimizador.load_state_dict(
            checkpoint["estado_otimizador"]
        )

        mover_estado_otimizador(
            otimizador,
            dispositivo,
        )

        print(
            "Estado do AdamW restaurado."
        )
    else:
        print(
            "AVISO: o checkpoint não possui o estado do "
            "otimizador. O AdamW será reiniciado."
        )

    passo_atual = int(
        checkpoint.get(
            "passo_atual",
            checkpoint.get("passos", 0),
        )
    )

    melhor_val_loss = float(
        checkpoint.get(
            "melhor_val_loss",
            math.inf,
        )
    )

    if "rng_python" in checkpoint:
        random.setstate(
            checkpoint["rng_python"]
        )

    if "rng_torch" in checkpoint:
        torch.set_rng_state(
            checkpoint["rng_torch"].cpu()
        )

    if (
        dispositivo.type == "cuda"
        and "rng_cuda" in checkpoint
    ):
        torch.cuda.set_rng_state_all(
            checkpoint["rng_cuda"]
        )

    print(
        f"Treinamento retomado a partir do passo "
        f"{passo_atual:,}."
    )

    return passo_atual, melhor_val_loss


def caminho_melhor_checkpoint(
    caminho_saida: Path,
) -> Path:
    """
    Cria um nome como qle_base_melhor.pt.
    """

    return caminho_saida.with_name(
        f"{caminho_saida.stem}_melhor"
        f"{caminho_saida.suffix}"
    )


def criar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treina um modelo Transformer causal inteiramente "
            "do zero."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config_qle_base.json"),
        help="Arquivo JSON de configuração.",
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus_QLE.txt"),
        help="Arquivo de texto usado no treinamento.",
    )

    parser.add_argument(
        "--saida",
        type=Path,
        default=Path(
            "checkpoints/qle_base_ultima.pt"
        ),
        help="Checkpoint mais recente.",
    )

    parser.add_argument(
        "--retomar",
        type=Path,
        default=None,
        help=(
            "Checkpoint existente para continuar o treinamento."
        ),
    )

    parser.add_argument(
        "--passos",
        type=int,
        default=None,
        help=(
            "Passo final desejado. Ao retomar do passo 300 "
            "com --passos 1500, serão executados mais 1200."
        ),
    )

    parser.add_argument(
        "--validacao-fracao",
        type=float,
        default=0.10,
        help="Fração do corpus reservada para validação.",
    )

    parser.add_argument(
        "--validar-cada",
        type=int,
        default=None,
        help="Intervalo entre avaliações.",
    )

    parser.add_argument(
        "--salvar-cada",
        type=int,
        default=None,
        help="Intervalo entre checkpoints.",
    )

    parser.add_argument(
        "--lotes-validacao",
        type=int,
        default=None,
        help="Quantidade máxima de lotes na validação.",
    )

    parser.add_argument(
        "--acumular-gradientes",
        type=int,
        default=None,
        help=(
            "Número de lotes acumulados antes de atualizar "
            "os pesos."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Processos auxiliares do DataLoader.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória.",
    )

    parser.add_argument(
        "--somente-avaliar",
        action="store_true",
        help="Carrega o checkpoint e executa somente validação.",
    )

    return parser.parse_args()


def main() -> None:
    args = criar_argumentos()

    configurar_sementes(args.seed)

    cfg_modelo, cfg_treino = carregar_config(
        args.config
    )

    validar_configuracao_treino(
        cfg_treino
    )

    if args.passos is not None:
        if args.passos <= 0:
            raise ValueError(
                "--passos precisa ser maior que zero."
            )

        cfg_treino["passos"] = args.passos

    validar_cada = int(
        args.validar_cada
        or cfg_treino.get("validar_cada", 100)
    )

    salvar_cada = int(
        args.salvar_cada
        or cfg_treino.get("salvar_cada", 100)
    )

    lotes_validacao = int(
        args.lotes_validacao
        or cfg_treino.get(
            "lotes_validacao",
            20,
        )
    )

    acumular_gradientes = int(
        args.acumular_gradientes
        or cfg_treino.get(
            "acumular_gradientes",
            1,
        )
    )

    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else cfg_treino.get(
            "num_workers",
            0,
        )
    )

    if validar_cada <= 0:
        raise ValueError(
            "validar_cada precisa ser maior que zero."
        )

    if salvar_cada <= 0:
        raise ValueError(
            "salvar_cada precisa ser maior que zero."
        )

    if lotes_validacao <= 0:
        raise ValueError(
            "lotes_validacao precisa ser maior que zero."
        )

    if acumular_gradientes <= 0:
        raise ValueError(
            "acumular_gradientes precisa ser maior que zero."
        )

    if not args.corpus.exists():
        raise FileNotFoundError(
            f"Corpus não encontrado: {args.corpus}"
        )

    texto = args.corpus.read_text(
        encoding="utf-8-sig"
    )

    if not texto.strip():
        raise ValueError(
            "O corpus está vazio."
        )

    verificar_texto_corrompido(texto)

    tokenizador = TokenizadorBytes()

    tokens = tokenizador.codificar(
        texto,
        adicionar_bos=True,
        adicionar_eos=True,
    )

    tokens_treino, tokens_validacao = dividir_tokens(
        tokens=tokens,
        comprimento_contexto=cfg_modelo.comprimento_max,
        fracao_validacao=args.validacao_fracao,
    )

    dispositivo = escolher_dispositivo()

    loader_treino = criar_loader(
        tokens=tokens_treino,
        comprimento_contexto=cfg_modelo.comprimento_max,
        batch_size=int(cfg_treino["batch_size"]),
        embaralhar=True,
        dispositivo=dispositivo,
        semente=args.seed,
        num_workers=num_workers,
    )

    loader_validacao = criar_loader(
        tokens=tokens_validacao,
        comprimento_contexto=cfg_modelo.comprimento_max,
        batch_size=int(cfg_treino["batch_size"]),
        embaralhar=False,
        dispositivo=dispositivo,
        semente=args.seed,
        num_workers=num_workers,
    )

    modelo = MiniLLM(
        cfg_modelo
    ).to(dispositivo)

    otimizador = AdamW(
        modelo.parameters(),
        lr=float(cfg_treino["lr_max"]),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=float(
            cfg_treino["weight_decay"]
        ),
    )

    passo_inicial = 0
    melhor_val_loss = math.inf

    if args.retomar is not None:
        passo_inicial, melhor_val_loss = (
            restaurar_checkpoint(
                caminho=args.retomar,
                modelo=modelo,
                otimizador=otimizador,
                configuracao_modelo=cfg_modelo,
                dispositivo=dispositivo,
            )
        )

    total_passos = int(
        cfg_treino["passos"]
    )

    if total_passos <= passo_inicial:
        if not args.somente_avaliar:
            raise ValueError(
                f"O checkpoint já está no passo {passo_inicial}, "
                f"mas o passo final solicitado é {total_passos}."
            )

    print("=" * 65)
    print("TREINAMENTO DA QLE")
    print("=" * 65)
    print(f"Dispositivo: {dispositivo}")
    print(
        f"Parâmetros: "
        f"{modelo.contar_parametros():,}"
    )
    print(
        f"Tokens totais: {len(tokens):,}"
    )
    print(
        f"Tokens de treino: "
        f"{len(tokens_treino):,}"
    )
    print(
        f"Tokens de validação: "
        f"{len(tokens_validacao):,}"
    )
    print(
        f"Contexto máximo: "
        f"{cfg_modelo.comprimento_max}"
    )
    print(
        f"Batch size: "
        f"{cfg_treino['batch_size']}"
    )
    print(
        f"Acúmulo de gradientes: "
        f"{acumular_gradientes}"
    )
    print(
        f"Batch efetivo: "
        f"{int(cfg_treino['batch_size']) * acumular_gradientes}"
    )
    print(
        f"Passo inicial: {passo_inicial:,}"
    )
    print(
        f"Passo final: {total_passos:,}"
    )
    print("=" * 65)

    if args.somente_avaliar:
        if args.retomar is None:
            raise ValueError(
                "--somente-avaliar exige --retomar."
            )

        val_loss = avaliar_modelo(
            modelo=modelo,
            loader=loader_validacao,
            dispositivo=dispositivo,
            max_lotes=lotes_validacao,
        )

        print(
            f"val_loss: {val_loss:.4f} | "
            f"val_ppl: "
            f"{perplexidade_segura(val_loss):.2f}"
        )

        return

    iterador = iter(
        loader_treino
    )

    inicio_total = time.perf_counter()
    inicio_intervalo = inicio_total

    modelo.train()

    melhor_caminho = caminho_melhor_checkpoint(
        args.saida
    )

    for passo in range(
        passo_inicial,
        total_passos,
    ):
        lr = taxa_aprendizado(
            passo=passo,
            total=total_passos,
            warmup=int(
                cfg_treino["warmup_passos"]
            ),
            lr_max=float(
                cfg_treino["lr_max"]
            ),
            lr_min=float(
                cfg_treino["lr_min"]
            ),
        )

        for grupo in otimizador.param_groups:
            grupo["lr"] = lr

        otimizador.zero_grad(
            set_to_none=True
        )

        perda_acumulada = 0.0

        for _ in range(
            acumular_gradientes
        ):
            iterador, x, y = proximo_lote(
                iterador,
                loader_treino,
            )

            x = x.to(
                dispositivo,
                non_blocking=True,
            )

            y = y.to(
                dispositivo,
                non_blocking=True,
            )

            _, perda = modelo(x, y)

            if perda is None:
                raise RuntimeError(
                    "O modelo não retornou loss."
                )

            perda_dividida = (
                perda
                / acumular_gradientes
            )

            perda_dividida.backward()

            perda_acumulada += float(
                perda.item()
            )

        norma_gradiente = clip_grad_norm_(
            modelo.parameters(),
            float(cfg_treino["grad_clip"]),
        )

        if not math.isfinite(
            float(norma_gradiente)
        ):
            raise FloatingPointError(
                "Gradiente não finito detectado. "
                "Reduza o learning rate."
            )

        otimizador.step()

        perda_media = (
            perda_acumulada
            / acumular_gradientes
        )

        passo_atual = passo + 1

        deve_logar = (
            passo == passo_inicial
            or passo_atual
            % int(cfg_treino["log_intervalo"])
            == 0
            or passo_atual == total_passos
        )

        if deve_logar:
            agora = time.perf_counter()
            tempo_intervalo = (
                agora - inicio_intervalo
            )

            tempo_total = (
                agora - inicio_total
            )

            print(
                f"passo {passo_atual:>6}/"
                f"{total_passos} | "
                f"loss {perda_media:.4f} | "
                f"ppl "
                f"{perplexidade_segura(perda_media):.2f} | "
                f"lr {lr:.2e} | "
                f"grad {float(norma_gradiente):.3f} | "
                f"{tempo_intervalo:.1f}s intervalo | "
                f"{tempo_total:.1f}s total"
            )

            inicio_intervalo = agora

        deve_validar = (
            passo_atual % validar_cada == 0
            or passo_atual == total_passos
        )

        if deve_validar:
            val_loss = avaliar_modelo(
                modelo=modelo,
                loader=loader_validacao,
                dispositivo=dispositivo,
                max_lotes=lotes_validacao,
            )

            val_ppl = perplexidade_segura(
                val_loss
            )

            print(
                f"VALIDAÇÃO | passo {passo_atual} | "
                f"val_loss {val_loss:.4f} | "
                f"val_ppl {val_ppl:.2f}"
            )

            if val_loss < melhor_val_loss:
                melhor_val_loss = val_loss

                salvar_checkpoint(
                    caminho=melhor_caminho,
                    modelo=modelo,
                    otimizador=otimizador,
                    configuracao_modelo=cfg_modelo,
                    configuracao_treino=cfg_treino,
                    passo_atual=passo_atual,
                    melhor_val_loss=melhor_val_loss,
                    tokens_treino=len(tokens_treino),
                    tokens_validacao=len(
                        tokens_validacao
                    ),
                )

                print(
                    "Novo melhor checkpoint salvo em: "
                    f"{melhor_caminho.resolve()}"
                )

        deve_salvar = (
            passo_atual % salvar_cada == 0
            or passo_atual == total_passos
        )

        if deve_salvar:
            salvar_checkpoint(
                caminho=args.saida,
                modelo=modelo,
                otimizador=otimizador,
                configuracao_modelo=cfg_modelo,
                configuracao_treino=cfg_treino,
                passo_atual=passo_atual,
                melhor_val_loss=melhor_val_loss,
                tokens_treino=len(tokens_treino),
                tokens_validacao=len(
                    tokens_validacao
                ),
            )

            print(
                "Checkpoint atual salvo em: "
                f"{args.saida.resolve()}"
            )

    tempo_final = (
        time.perf_counter()
        - inicio_total
    )

    print("\n" + "=" * 65)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 65)
    print(
        f"Passos realizados nesta execução: "
        f"{total_passos - passo_inicial:,}"
    )
    print(
        f"Passo total atingido: "
        f"{total_passos:,}"
    )
    print(
        f"Melhor val_loss: "
        f"{melhor_val_loss:.4f}"
    )
    print(
        f"Tempo total: "
        f"{tempo_final:.1f} segundos"
    )
    print(
        f"Último checkpoint: "
        f"{args.saida.resolve()}"
    )
    print(
        f"Melhor checkpoint: "
        f"{melhor_caminho.resolve()}"
    )


if __name__ == "__main__":
    main()