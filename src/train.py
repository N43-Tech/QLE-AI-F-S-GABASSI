from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset import DatasetDialogosBPE
from src.model import ConfigModelo, MiniLLM
from src.tokenizer_bpe import TokenizadorBPE


FORMATO_CHECKPOINT = "qle_bpe_v1"
VERSAO_CHECKPOINT = 1
NOME_IA = "QLE"


class ErroTreinamento(RuntimeError):
    """Erro controlado do pipeline de treinamento da QLE."""


class AgendadorWarmupCoseno:
    """
    Warmup linear seguido de decaimento cosseno.

    O estado salvo contém o número de atualizações já realizadas.
    O próximo learning rate é calculado com base nesse número.
    """

    def __init__(
        self,
        otimizador: AdamW,
        *,
        total_passos: int,
        warmup_passos: int,
        lr_max: float,
        lr_min: float,
    ) -> None:
        if total_passos <= 0:
            raise ValueError("total_passos precisa ser positivo.")

        if not 0 <= warmup_passos < total_passos:
            raise ValueError(
                "warmup_passos precisa estar entre 0 e total_passos - 1."
            )

        if lr_max <= 0:
            raise ValueError("lr_max precisa ser positivo.")

        if not 0 <= lr_min <= lr_max:
            raise ValueError(
                "lr_min precisa estar entre 0 e lr_max."
            )

        self.otimizador = otimizador
        self.total_passos = int(total_passos)
        self.warmup_passos = int(warmup_passos)
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        self.passo_atual = 0

    def _calcular_lr(
        self,
        indice_passo: int,
    ) -> float:
        if self.warmup_passos > 0 and indice_passo < self.warmup_passos:
            return (
                self.lr_max
                * (indice_passo + 1)
                / self.warmup_passos
            )

        inicio_coseno = self.warmup_passos
        quantidade_coseno = max(
            1,
            self.total_passos - inicio_coseno - 1,
        )

        progresso = (
            indice_passo - inicio_coseno
        ) / quantidade_coseno

        progresso = min(
            max(progresso, 0.0),
            1.0,
        )

        fator = 0.5 * (
            1.0 + math.cos(math.pi * progresso)
        )

        return (
            self.lr_min
            + (self.lr_max - self.lr_min) * fator
        )

    def preparar_proximo_passo(self) -> float:
        lr = self._calcular_lr(
            self.passo_atual
        )

        for grupo in self.otimizador.param_groups:
            grupo["lr"] = lr

        return lr

    def registrar_passo(self) -> None:
        self.passo_atual += 1

    def get_last_lr(self) -> list[float]:
        return [
            float(grupo["lr"])
            for grupo in self.otimizador.param_groups
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "passo_atual": self.passo_atual,
            "total_passos": self.total_passos,
            "warmup_passos": self.warmup_passos,
            "lr_max": self.lr_max,
            "lr_min": self.lr_min,
        }

    def load_state_dict(
        self,
        estado: dict[str, Any],
    ) -> None:
        passo = int(
            estado.get("passo_atual", 0)
        )

        if not 0 <= passo <= self.total_passos:
            raise ValueError(
                f"Passo inválido no agendador: {passo}."
            )

        self.passo_atual = passo


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
            raise ErroTreinamento(
                "CUDA foi solicitada, mas não está disponível."
            )
        return torch.device("cuda")

    if solicitado == "mps":
        disponivel = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

        if not disponivel:
            raise ErroTreinamento(
                "MPS foi solicitado, mas não está disponível."
            )
        return torch.device("mps")

    if solicitado != "auto":
        raise ErroTreinamento(
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


def configurar_sementes(
    semente: int,
    deterministico: bool,
) -> None:
    random.seed(semente)
    torch.manual_seed(semente)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)

    if deterministico:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

        if hasattr(
            torch.backends,
            "cudnn",
        ):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def inicializar_worker(
    worker_id: int,
    *,
    semente_base: int,
) -> None:
    semente = semente_base + worker_id
    random.seed(semente)
    torch.manual_seed(semente)


def agora_utc_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")


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


def dados_arquivo(
    caminho: Path,
) -> dict[str, Any]:
    return {
        "caminho": str(
            caminho.resolve()
        ),
        "nome": caminho.name,
        "tamanho_bytes": (
            caminho.stat().st_size
        ),
        "sha256": sha256_arquivo(
            caminho
        ),
    }


def carregar_config(
    caminho: Path,
) -> tuple[
    ConfigModelo,
    dict[str, Any],
]:
    if not caminho.exists():
        raise ErroTreinamento(
            "Arquivo de configuração não encontrado: "
            f"{caminho.resolve()}"
        )

    try:
        dados = json.loads(
            caminho.read_text(
                encoding="utf-8-sig"
            )
        )
    except json.JSONDecodeError as erro:
        raise ErroTreinamento(
            f"JSON inválido em {caminho}: {erro}"
        ) from erro

    if not isinstance(dados, dict):
        raise ErroTreinamento(
            "A configuração principal precisa ser um objeto JSON."
        )

    if "modelo" not in dados:
        raise ErroTreinamento(
            "A configuração não possui a seção 'modelo'."
        )

    if "treino" not in dados:
        raise ErroTreinamento(
            "A configuração não possui a seção 'treino'."
        )

    cfg_modelo = ConfigModelo(
        **dados["modelo"]
    )
    cfg_modelo.validar()

    cfg_treino = dict(
        dados["treino"]
    )

    return cfg_modelo, cfg_treino


def valor_config(
    configuracao: dict[str, Any],
    nome: str,
    padrao: Any,
) -> Any:
    return configuracao.get(
        nome,
        padrao,
    )


def validar_config_treino(
    cfg: dict[str, Any],
) -> None:
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
        chave
        for chave in obrigatorios
        if chave not in cfg
    ]

    if ausentes:
        raise ErroTreinamento(
            "Campos ausentes na configuração de treino: "
            + ", ".join(ausentes)
        )

    inteiros_positivos = (
        "batch_size",
        "passos",
        "log_intervalo",
    )

    for nome in inteiros_positivos:
        if int(cfg[nome]) <= 0:
            raise ErroTreinamento(
                f"{nome} precisa ser maior que zero."
            )

    passos = int(cfg["passos"])
    warmup = int(cfg["warmup_passos"])

    if not 0 <= warmup < passos:
        raise ErroTreinamento(
            "warmup_passos precisa estar entre 0 e passos - 1."
        )

    lr_max = float(cfg["lr_max"])
    lr_min = float(cfg["lr_min"])

    if lr_max <= 0:
        raise ErroTreinamento(
            "lr_max precisa ser positivo."
        )

    if not 0 <= lr_min <= lr_max:
        raise ErroTreinamento(
            "lr_min precisa estar entre 0 e lr_max."
        )

    if float(cfg["weight_decay"]) < 0:
        raise ErroTreinamento(
            "weight_decay não pode ser negativo."
        )

    if float(cfg["grad_clip"]) <= 0:
        raise ErroTreinamento(
            "grad_clip precisa ser positivo."
        )


def criar_loader(
    dataset: DatasetDialogosBPE,
    *,
    batch_size: int,
    embaralhar: bool,
    dispositivo: torch.device,
    semente: int,
    num_workers: int,
) -> DataLoader:
    if len(dataset) == 0:
        raise ErroTreinamento(
            "O dataset está vazio."
        )

    batch_real = min(
        batch_size,
        len(dataset),
    )

    gerador = torch.Generator()
    gerador.manual_seed(semente)

    worker_init = partial(
        inicializar_worker,
        semente_base=semente,
    )

    return DataLoader(
        dataset,
        batch_size=batch_real,
        shuffle=embaralhar,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(
            dispositivo.type == "cuda"
        ),
        persistent_workers=(
            num_workers > 0
        ),
        generator=gerador,
        worker_init_fn=worker_init,
    )


def proximo_lote(
    iterador: Any,
    loader: DataLoader,
) -> tuple[
    Any,
    torch.Tensor,
    torch.Tensor,
]:
    try:
        x, y = next(iterador)
    except StopIteration:
        iterador = iter(loader)
        x, y = next(iterador)

    return iterador, x, y


def perplexidade_segura(
    perda: float,
) -> float:
    return math.exp(
        min(perda, 20.0)
    )


def avaliar_modelo(
    modelo: MiniLLM,
    loader: DataLoader,
    dispositivo: torch.device,
    max_lotes: int,
) -> tuple[float, int]:
    modelo.eval()

    soma_perda = 0.0
    quantidade_tokens = 0
    lotes_avaliados = 0

    with torch.inference_mode():
        for indice, (x, y) in enumerate(
            loader
        ):
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

            _, perda = modelo(
                x,
                y,
            )

            if perda is None:
                raise ErroTreinamento(
                    "O modelo não retornou loss na validação."
                )

            perda_valor = float(
                perda.item()
            )

            if not math.isfinite(
                perda_valor
            ):
                raise FloatingPointError(
                    "Loss não finita detectada na validação."
                )

            tokens_validos = int(
                (y != -100)
                .sum()
                .item()
            )

            if tokens_validos <= 0:
                continue

            soma_perda += (
                perda_valor
                * tokens_validos
            )
            quantidade_tokens += tokens_validos
            lotes_avaliados += 1

    modelo.train()

    if quantidade_tokens <= 0:
        raise ErroTreinamento(
            "Nenhum token válido foi avaliado."
        )

    return (
        soma_perda / quantidade_tokens,
        lotes_avaliados,
    )


def mover_estado_otimizador(
    otimizador: AdamW,
    dispositivo: torch.device,
) -> None:
    for estado in otimizador.state.values():
        for chave, valor in estado.items():
            if isinstance(
                valor,
                torch.Tensor,
            ):
                estado[chave] = valor.to(
                    dispositivo
                )


def serializar_tokens_especiais(
    tokenizador: TokenizadorBPE,
) -> dict[str, int]:
    return {
        chave: int(valor)
        for chave, valor in asdict(
            tokenizador.especiais
        ).items()
    }


def escrever_log_jsonl(
    caminho: Path,
    evento: dict[str, Any],
) -> None:
    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with caminho.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as arquivo:
        arquivo.write(
            json.dumps(
                evento,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


def salvar_checkpoint(
    *,
    caminho: Path,
    modelo: MiniLLM,
    otimizador: AdamW,
    agendador: AgendadorWarmupCoseno,
    cfg_modelo: ConfigModelo,
    cfg_treino: dict[str, Any],
    passo_atual: int,
    melhor_val_loss: float,
    ultima_train_loss: float | None,
    ultima_val_loss: float | None,
    tokenizador: TokenizadorBPE,
    caminho_tokenizador: Path,
    dataset_treino: DatasetDialogosBPE,
    dataset_validacao: DatasetDialogosBPE,
    caminho_treino: Path,
    caminho_validacao: Path,
) -> None:
    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporario = caminho.with_name(
        caminho.name + ".tmp"
    )

    dados: dict[str, Any] = {
        "formato_checkpoint": FORMATO_CHECKPOINT,
        "versao_checkpoint": VERSAO_CHECKPOINT,
        "nome_ia": NOME_IA,
        "criado_em_utc": agora_utc_iso(),
        "config_modelo": (
            cfg_modelo.para_dict()
        ),
        "config_treino": cfg_treino,
        "estado_modelo": (
            modelo.state_dict()
        ),
        "estado_otimizador": (
            otimizador.state_dict()
        ),
        "estado_agendador": (
            agendador.state_dict()
        ),
        "passo_atual": passo_atual,
        "passos": passo_atual,
        "melhor_val_loss": (
            melhor_val_loss
        ),
        "metricas": {
            "ultima_train_loss": (
                ultima_train_loss
            ),
            "ultima_val_loss": (
                ultima_val_loss
            ),
        },
        "tokenizador": {
            "tipo": "bpe",
            **dados_arquivo(
                caminho_tokenizador
            ),
            "tamanho_vocab": (
                tokenizador.tamanho_vocab
            ),
            "tokens_especiais": (
                serializar_tokens_especiais(
                    tokenizador
                )
            ),
        },
        "datasets": {
            "treino": {
                **dados_arquivo(
                    caminho_treino
                ),
                "exemplos": len(
                    dataset_treino
                ),
            },
            "validacao": {
                **dados_arquivo(
                    caminho_validacao
                ),
                "exemplos": len(
                    dataset_validacao
                ),
            },
        },
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

    os.replace(
        temporario,
        caminho,
    )


def validar_checkpoint_retomada(
    checkpoint: dict[str, Any],
    *,
    cfg_modelo: ConfigModelo,
    caminho_tokenizador: Path,
    caminho_treino: Path,
    caminho_validacao: Path,
    permitir_dataset_alterado: bool,
) -> None:
    formato = checkpoint.get(
        "formato_checkpoint"
    )

    if formato != FORMATO_CHECKPOINT:
        raise ErroTreinamento(
            "O checkpoint não é compatível com o pipeline BPE atual. "
            f"Formato encontrado: {formato!r}."
        )

    config_salva = checkpoint.get(
        "config_modelo"
    )

    if config_salva != cfg_modelo.para_dict():
        raise ErroTreinamento(
            "A arquitetura do checkpoint é diferente da configuração atual."
        )

    meta_tokenizador = checkpoint.get(
        "tokenizador",
        {},
    )

    hash_salvo = meta_tokenizador.get(
        "sha256"
    )
    hash_atual = sha256_arquivo(
        caminho_tokenizador
    )

    if hash_salvo and hash_salvo != hash_atual:
        raise ErroTreinamento(
            "O tokenizador atual é diferente do tokenizador "
            "utilizado pelo checkpoint."
        )

    datasets_salvos = checkpoint.get(
        "datasets",
        {},
    )

    alteracoes: list[str] = []

    for nome, caminho in (
        ("treino", caminho_treino),
        ("validacao", caminho_validacao),
    ):
        hash_anterior = (
            datasets_salvos
            .get(nome, {})
            .get("sha256")
        )

        if (
            hash_anterior
            and hash_anterior
            != sha256_arquivo(caminho)
        ):
            alteracoes.append(nome)

    if alteracoes and not permitir_dataset_alterado:
        raise ErroTreinamento(
            "O dataset foi alterado desde o checkpoint: "
            + ", ".join(alteracoes)
            + ". Use --permitir-dataset-alterado somente se essa "
            "mudança for intencional."
        )


def restaurar_checkpoint(
    *,
    caminho: Path,
    modelo: MiniLLM,
    otimizador: AdamW,
    agendador: AgendadorWarmupCoseno,
    cfg_modelo: ConfigModelo,
    dispositivo: torch.device,
    caminho_tokenizador: Path,
    caminho_treino: Path,
    caminho_validacao: Path,
    permitir_dataset_alterado: bool,
) -> tuple[
    int,
    float,
    float | None,
    float | None,
]:
    if not caminho.exists():
        raise ErroTreinamento(
            "Checkpoint para retomada não encontrado: "
            f"{caminho.resolve()}"
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
        raise ErroTreinamento(
            "Não foi possível carregar o checkpoint. "
            f"Tamanho: {caminho.stat().st_size} bytes. "
            "O arquivo pode estar corrompido."
        ) from erro

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ErroTreinamento(
            "O checkpoint não é um dicionário válido."
        )

    validar_checkpoint_retomada(
        checkpoint,
        cfg_modelo=cfg_modelo,
        caminho_tokenizador=(
            caminho_tokenizador
        ),
        caminho_treino=caminho_treino,
        caminho_validacao=(
            caminho_validacao
        ),
        permitir_dataset_alterado=(
            permitir_dataset_alterado
        ),
    )

    modelo.load_state_dict(
        checkpoint["estado_modelo"],
        strict=True,
    )

    if "estado_otimizador" not in checkpoint:
        raise ErroTreinamento(
            "O checkpoint não possui estado do otimizador."
        )

    otimizador.load_state_dict(
        checkpoint["estado_otimizador"]
    )
    mover_estado_otimizador(
        otimizador,
        dispositivo,
    )

    if "estado_agendador" in checkpoint:
        agendador.load_state_dict(
            checkpoint[
                "estado_agendador"
            ]
        )
    else:
        agendador.passo_atual = int(
            checkpoint.get(
                "passo_atual",
                0,
            )
        )

    passo_atual = int(
        checkpoint.get(
            "passo_atual",
            0,
        )
    )

    if agendador.passo_atual != passo_atual:
        agendador.passo_atual = passo_atual

    melhor_val_loss = float(
        checkpoint.get(
            "melhor_val_loss",
            math.inf,
        )
    )

    metricas = checkpoint.get(
        "metricas",
        {},
    )

    ultima_train_loss = metricas.get(
        "ultima_train_loss"
    )
    ultima_val_loss = metricas.get(
        "ultima_val_loss"
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

    return (
        passo_atual,
        melhor_val_loss,
        ultima_train_loss,
        ultima_val_loss,
    )


def validar_argumentos_numericos(
    *,
    validar_cada: int,
    salvar_cada: int,
    lotes_validacao: int,
    acumular_gradientes: int,
    num_workers: int,
) -> None:
    valores_positivos = {
        "validar_cada": validar_cada,
        "salvar_cada": salvar_cada,
        "lotes_validacao": lotes_validacao,
        "acumular_gradientes": (
            acumular_gradientes
        ),
    }

    for nome, valor in valores_positivos.items():
        if valor <= 0:
            raise ErroTreinamento(
                f"{nome} precisa ser positivo."
            )

    if num_workers < 0:
        raise ErroTreinamento(
            "num_workers não pode ser negativo."
        )


def criar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treina a QLE com tokenizador BPE e loss "
            "supervisionada somente sobre as respostas."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config_qle_bpe_smoke.json"
        ),
    )

    parser.add_argument(
        "--tokenizador",
        type=Path,
        default=Path(
            "tokenizer/qle_bpe_2000.json"
        ),
    )

    parser.add_argument(
        "--treino",
        type=Path,
        default=Path(
            "data/splits/train.jsonl"
        ),
    )

    parser.add_argument(
        "--validacao",
        type=Path,
        default=Path(
            "data/splits/validation.jsonl"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        "--saida",
        dest="checkpoint",
        type=Path,
        default=Path(
            "checkpoints/qle_bpe_ultimo.pt"
        ),
    )

    parser.add_argument(
        "--melhor-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/qle_bpe_melhor.pt"
        ),
    )

    parser.add_argument(
        "--retomar",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--log",
        type=Path,
        default=Path(
            "logs/treino_qle_bpe.jsonl"
        ),
    )

    parser.add_argument(
        "--passos",
        type=int,
        default=None,
        help="Sobrescreve o passo final definido no JSON.",
    )

    parser.add_argument(
        "--validar-cada",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--salvar-cada",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--lotes-validacao",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--acumular-gradientes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
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
        "--deterministico",
        action="store_true",
    )

    parser.add_argument(
        "--permitir-dataset-alterado",
        action="store_true",
    )

    parser.add_argument(
        "--somente-avaliar",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    configurar_utf8()
    args = criar_argumentos()

    configurar_sementes(
        args.seed,
        args.deterministico,
    )

    cfg_modelo, cfg_treino = carregar_config(
        args.config
    )

    if args.passos is not None:
        if args.passos <= 0:
            raise ErroTreinamento(
                "--passos precisa ser positivo."
            )

        cfg_treino["passos"] = (
            args.passos
        )

        if (
            int(
                cfg_treino[
                    "warmup_passos"
                ]
            )
            >= args.passos
        ):
            cfg_treino[
                "warmup_passos"
            ] = max(
                0,
                args.passos // 10,
            )

    validar_config_treino(
        cfg_treino
    )

    validar_cada = int(
        args.validar_cada
        if args.validar_cada is not None
        else valor_config(
            cfg_treino,
            "validar_cada",
            100,
        )
    )

    salvar_cada = int(
        args.salvar_cada
        if args.salvar_cada is not None
        else valor_config(
            cfg_treino,
            "salvar_cada",
            100,
        )
    )

    lotes_validacao = int(
        args.lotes_validacao
        if args.lotes_validacao is not None
        else valor_config(
            cfg_treino,
            "lotes_validacao",
            30,
        )
    )

    acumular_gradientes = int(
        args.acumular_gradientes
        if args.acumular_gradientes is not None
        else valor_config(
            cfg_treino,
            "acumular_gradientes",
            1,
        )
    )

    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else valor_config(
            cfg_treino,
            "num_workers",
            0,
        )
    )

    validar_argumentos_numericos(
        validar_cada=validar_cada,
        salvar_cada=salvar_cada,
        lotes_validacao=lotes_validacao,
        acumular_gradientes=(
            acumular_gradientes
        ),
        num_workers=num_workers,
    )

    tokenizador = TokenizadorBPE(
        args.tokenizador
    )

    if (
        tokenizador.tamanho_vocab
        != cfg_modelo.tamanho_vocab
    ):
        raise ErroTreinamento(
            "O tamanho do vocabulário do tokenizador "
            f"({tokenizador.tamanho_vocab}) difere da configuração "
            f"do modelo ({cfg_modelo.tamanho_vocab})."
        )

    dataset_treino = DatasetDialogosBPE(
        args.treino,
        tokenizador,
        cfg_modelo.comprimento_max,
    )

    dataset_validacao = DatasetDialogosBPE(
        args.validacao,
        tokenizador,
        cfg_modelo.comprimento_max,
    )

    dispositivo = escolher_dispositivo(
        args.dispositivo
    )

    loader_treino = criar_loader(
        dataset_treino,
        batch_size=int(
            cfg_treino["batch_size"]
        ),
        embaralhar=True,
        dispositivo=dispositivo,
        semente=args.seed,
        num_workers=num_workers,
    )

    loader_validacao = criar_loader(
        dataset_validacao,
        batch_size=int(
            cfg_treino["batch_size"]
        ),
        embaralhar=False,
        dispositivo=dispositivo,
        semente=args.seed + 10_000,
        num_workers=num_workers,
    )

    modelo = MiniLLM(
        cfg_modelo
    ).to(dispositivo)

    otimizador = AdamW(
        modelo.parameters(),
        lr=float(
            cfg_treino["lr_max"]
        ),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=float(
            cfg_treino["weight_decay"]
        ),
    )

    total_passos = int(
        cfg_treino["passos"]
    )

    agendador = AgendadorWarmupCoseno(
        otimizador,
        total_passos=total_passos,
        warmup_passos=int(
            cfg_treino[
                "warmup_passos"
            ]
        ),
        lr_max=float(
            cfg_treino["lr_max"]
        ),
        lr_min=float(
            cfg_treino["lr_min"]
        ),
    )

    passo_inicial = 0
    melhor_val_loss = math.inf
    ultima_train_loss: float | None = None
    ultima_val_loss: float | None = None

    if args.retomar is not None:
        (
            passo_inicial,
            melhor_val_loss,
            ultima_train_loss,
            ultima_val_loss,
        ) = restaurar_checkpoint(
            caminho=args.retomar,
            modelo=modelo,
            otimizador=otimizador,
            agendador=agendador,
            cfg_modelo=cfg_modelo,
            dispositivo=dispositivo,
            caminho_tokenizador=(
                args.tokenizador
            ),
            caminho_treino=args.treino,
            caminho_validacao=(
                args.validacao
            ),
            permitir_dataset_alterado=(
                args.permitir_dataset_alterado
            ),
        )

    if passo_inicial > total_passos:
        raise ErroTreinamento(
            f"O checkpoint está no passo {passo_inicial}, "
            f"acima do passo final {total_passos}."
        )

    print("=" * 72)
    print("TREINAMENTO SUPERVISIONADO DA QLE — BPE")
    print("=" * 72)
    print(f"Dispositivo: {dispositivo}")
    print(
        f"Parâmetros: "
        f"{modelo.contar_parametros():,}"
    )
    print(
        f"Vocabulário: "
        f"{tokenizador.tamanho_vocab:,}"
    )
    print(
        f"Treino: "
        f"{len(dataset_treino):,} exemplos"
    )
    print(
        f"Validação: "
        f"{len(dataset_validacao):,} exemplos"
    )
    print(
        f"Contexto: "
        f"{cfg_modelo.comprimento_max}"
    )
    print(
        f"Batch size: "
        f"{cfg_treino['batch_size']}"
    )
    print(
        f"Acumulação: "
        f"{acumular_gradientes}"
    )
    print(
        "Batch efetivo: "
        f"{int(cfg_treino['batch_size']) * acumular_gradientes}"
    )
    print(
        f"Passo inicial: {passo_inicial:,}"
    )
    print(
        f"Passo final: {total_passos:,}"
    )
    print("=" * 72)

    if args.somente_avaliar:
        if args.retomar is None:
            raise ErroTreinamento(
                "--somente-avaliar exige --retomar."
            )

        val_loss, lotes = avaliar_modelo(
            modelo,
            loader_validacao,
            dispositivo,
            lotes_validacao,
        )

        print(
            f"val_loss: {val_loss:.4f} | "
            f"val_ppl: {perplexidade_segura(val_loss):.2f} | "
            f"lotes: {lotes}"
        )
        return

    if passo_inicial == total_passos:
        raise ErroTreinamento(
            "O checkpoint já atingiu o passo final solicitado."
        )

    iterador = iter(
        loader_treino
    )

    inicio_total = time.perf_counter()
    inicio_intervalo = inicio_total
    modelo.train()

    escrever_log_jsonl(
        args.log,
        {
            "evento": "inicio",
            "momento_utc": agora_utc_iso(),
            "passo_inicial": passo_inicial,
            "passo_final": total_passos,
            "dispositivo": str(
                dispositivo
            ),
            "parametros": (
                modelo.contar_parametros()
            ),
            "exemplos_treino": len(
                dataset_treino
            ),
            "exemplos_validacao": len(
                dataset_validacao
            ),
        },
    )

    for passo_atual in range(
        passo_inicial,
        total_passos,
    ):
        lr = (
            agendador
            .preparar_proximo_passo()
        )

        otimizador.zero_grad(
            set_to_none=True
        )

        soma_perdas = 0.0

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

            _, perda = modelo(
                x,
                y,
            )

            if perda is None:
                raise ErroTreinamento(
                    "O modelo não retornou loss."
                )

            perda_valor = float(
                perda.item()
            )

            if not math.isfinite(
                perda_valor
            ):
                raise FloatingPointError(
                    "Loss não finita detectada. "
                    "Reduza o learning rate e valide os dados."
                )

            (
                perda
                / acumular_gradientes
            ).backward()

            soma_perdas += perda_valor

        norma_gradiente = clip_grad_norm_(
            modelo.parameters(),
            float(
                cfg_treino[
                    "grad_clip"
                ]
            ),
        )

        norma_valor = float(
            norma_gradiente
        )

        if not math.isfinite(
            norma_valor
        ):
            raise FloatingPointError(
                "Gradiente não finito detectado."
            )

        otimizador.step()
        agendador.registrar_passo()

        numero_passo = passo_atual + 1
        ultima_train_loss = (
            soma_perdas
            / acumular_gradientes
        )

        deve_logar = (
            numero_passo == passo_inicial + 1
            or numero_passo
            % int(
                cfg_treino[
                    "log_intervalo"
                ]
            )
            == 0
            or numero_passo == total_passos
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
                f"passo {numero_passo:>6}/"
                f"{total_passos} | "
                f"loss {ultima_train_loss:.4f} | "
                f"ppl "
                f"{perplexidade_segura(ultima_train_loss):.2f} | "
                f"lr {lr:.2e} | "
                f"grad {norma_valor:.3f} | "
                f"{tempo_intervalo:.1f}s intervalo | "
                f"{tempo_total:.1f}s total"
            )

            escrever_log_jsonl(
                args.log,
                {
                    "evento": "treino",
                    "momento_utc": agora_utc_iso(),
                    "passo": numero_passo,
                    "train_loss": (
                        ultima_train_loss
                    ),
                    "train_ppl": (
                        perplexidade_segura(
                            ultima_train_loss
                        )
                    ),
                    "learning_rate": lr,
                    "grad_norm": (
                        norma_valor
                    ),
                    "tempo_total_segundos": (
                        tempo_total
                    ),
                },
            )

            inicio_intervalo = agora

        deve_validar = (
            numero_passo
            % validar_cada
            == 0
            or numero_passo
            == total_passos
        )

        if deve_validar:
            ultima_val_loss, lotes = avaliar_modelo(
                modelo,
                loader_validacao,
                dispositivo,
                lotes_validacao,
            )

            val_ppl = perplexidade_segura(
                ultima_val_loss
            )

            print(
                f"VALIDAÇÃO | passo {numero_passo} | "
                f"val_loss {ultima_val_loss:.4f} | "
                f"val_ppl {val_ppl:.2f} | "
                f"lotes {lotes}"
            )

            escrever_log_jsonl(
                args.log,
                {
                    "evento": "validacao",
                    "momento_utc": agora_utc_iso(),
                    "passo": numero_passo,
                    "validation_loss": (
                        ultima_val_loss
                    ),
                    "validation_ppl": (
                        val_ppl
                    ),
                    "lotes": lotes,
                },
            )

            if (
                ultima_val_loss
                < melhor_val_loss
            ):
                melhor_val_loss = (
                    ultima_val_loss
                )

                salvar_checkpoint(
                    caminho=(
                        args.melhor_checkpoint
                    ),
                    modelo=modelo,
                    otimizador=otimizador,
                    agendador=agendador,
                    cfg_modelo=cfg_modelo,
                    cfg_treino=cfg_treino,
                    passo_atual=numero_passo,
                    melhor_val_loss=(
                        melhor_val_loss
                    ),
                    ultima_train_loss=(
                        ultima_train_loss
                    ),
                    ultima_val_loss=(
                        ultima_val_loss
                    ),
                    tokenizador=tokenizador,
                    caminho_tokenizador=(
                        args.tokenizador
                    ),
                    dataset_treino=(
                        dataset_treino
                    ),
                    dataset_validacao=(
                        dataset_validacao
                    ),
                    caminho_treino=(
                        args.treino
                    ),
                    caminho_validacao=(
                        args.validacao
                    ),
                )

                print(
                    "Novo melhor checkpoint: "
                    f"{args.melhor_checkpoint.resolve()}"
                )

        deve_salvar = (
            numero_passo
            % salvar_cada
            == 0
            or numero_passo
            == total_passos
        )

        if deve_salvar:
            salvar_checkpoint(
                caminho=args.checkpoint,
                modelo=modelo,
                otimizador=otimizador,
                agendador=agendador,
                cfg_modelo=cfg_modelo,
                cfg_treino=cfg_treino,
                passo_atual=numero_passo,
                melhor_val_loss=(
                    melhor_val_loss
                ),
                ultima_train_loss=(
                    ultima_train_loss
                ),
                ultima_val_loss=(
                    ultima_val_loss
                ),
                tokenizador=tokenizador,
                caminho_tokenizador=(
                    args.tokenizador
                ),
                dataset_treino=(
                    dataset_treino
                ),
                dataset_validacao=(
                    dataset_validacao
                ),
                caminho_treino=(
                    args.treino
                ),
                caminho_validacao=(
                    args.validacao
                ),
            )

            print(
                "Checkpoint atual: "
                f"{args.checkpoint.resolve()}"
            )

    tempo_total = (
        time.perf_counter()
        - inicio_total
    )

    print("\n" + "=" * 72)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 72)
    print(
        f"Passo total atingido: "
        f"{total_passos:,}"
    )
    print(
        f"Última train_loss: "
        f"{ultima_train_loss:.4f}"
    )

    if ultima_val_loss is not None:
        print(
            f"Última val_loss: "
            f"{ultima_val_loss:.4f}"
        )

    print(
        f"Melhor val_loss: "
        f"{melhor_val_loss:.4f}"
    )
    print(
        f"Tempo total: "
        f"{tempo_total:.1f} segundos"
    )
    print(
        f"Checkpoint atual: "
        f"{args.checkpoint.resolve()}"
    )
    print(
        f"Melhor checkpoint: "
        f"{args.melhor_checkpoint.resolve()}"
    )


if __name__ == "__main__":
    try:
        main()
    except ErroTreinamento as erro:
        print(
            f"ERRO: {erro}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
