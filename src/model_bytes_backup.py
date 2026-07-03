from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConfigModelo:
    tamanho_vocab: int = 260
    d_model: int = 128
    n_camadas: int = 4
    n_cabecas: int = 4
    n_cabecas_kv: int = 2
    d_ff: int = 384
    comprimento_max: int = 128
    dropout: float = 0.0
    rope_theta: float = 10_000.0

    def validar(self) -> None:
        if self.d_model % self.n_cabecas != 0:
            raise ValueError("d_model deve ser divisível por n_cabecas.")
        if self.n_cabecas % self.n_cabecas_kv != 0:
            raise ValueError("n_cabecas deve ser divisível por n_cabecas_kv.")
        if (self.d_model // self.n_cabecas) % 2 != 0:
            raise ValueError("A dimensão de cada cabeça deve ser par para RoPE.")

    def para_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dimensao: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.peso = nn.Parameter(torch.ones(dimensao))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_inverso = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms_inverso).to(x.dtype) * self.peso


class RoPE(nn.Module):
    def __init__(self, dimensao_cabeca: int, comprimento_max: int, theta: float) -> None:
        super().__init__()
        frequencias = 1.0 / (
            theta ** (torch.arange(0, dimensao_cabeca, 2, dtype=torch.float32) / dimensao_cabeca)
        )
        posicoes = torch.arange(comprimento_max, dtype=torch.float32)
        angulos = torch.outer(posicoes, frequencias)
        self.register_buffer("cos", angulos.cos(), persistent=False)
        self.register_buffer("sin", angulos.sin(), persistent=False)

    def aplicar(self, x: torch.Tensor, inicio: int = 0) -> torch.Tensor:
        # x: (B, H, T, D)
        tamanho = x.size(-2)
        cos = self.cos[inicio : inicio + tamanho].view(1, 1, tamanho, -1)
        sin = self.sin[inicio : inicio + tamanho].view(1, 1, tamanho, -1)
        pares = x.float().reshape(*x.shape[:-1], -1, 2)
        x_par = pares[..., 0]
        x_impar = pares[..., 1]
        rot_par = x_par * cos - x_impar * sin
        rot_impar = x_par * sin + x_impar * cos
        return torch.stack((rot_par, rot_impar), dim=-1).flatten(-2).to(x.dtype)


class AtencaoCausalGQA(nn.Module):
    def __init__(self, cfg: ConfigModelo) -> None:
        super().__init__()
        self.n_cabecas = cfg.n_cabecas
        self.n_cabecas_kv = cfg.n_cabecas_kv
        self.d_cabeca = cfg.d_model // cfg.n_cabecas
        self.repeticoes_kv = cfg.n_cabecas // cfg.n_cabecas_kv
        self.q = nn.Linear(cfg.d_model, cfg.n_cabecas * self.d_cabeca, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_cabecas_kv * self.d_cabeca, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_cabecas_kv * self.d_cabeca, bias=False)
        self.saida = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.rope = RoPE(self.d_cabeca, cfg.comprimento_max, cfg.rope_theta)

    def _separar(self, x: torch.Tensor, n_cabecas: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, n_cabecas, self.d_cabeca).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._separar(self.q(x), self.n_cabecas)
        k = self._separar(self.k(x), self.n_cabecas_kv)
        v = self._separar(self.v(x), self.n_cabecas_kv)
        q = self.rope.aplicar(q)
        k = self.rope.aplicar(k)
        k = k.repeat_interleave(self.repeticoes_kv, dim=1)
        v = v.repeat_interleave(self.repeticoes_kv, dim=1)
        contexto = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        b, _, t, _ = contexto.shape
        contexto = contexto.transpose(1, 2).contiguous().view(b, t, -1)
        return self.saida(contexto)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ConfigModelo) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class BlocoTransformer(nn.Module):
    def __init__(self, cfg: ConfigModelo) -> None:
        super().__init__()
        self.norma_atencao = RMSNorm(cfg.d_model)
        self.atencao = AtencaoCausalGQA(cfg)
        self.norma_ffn = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.atencao(self.norma_atencao(x))
        x = x + self.ffn(self.norma_ffn(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self, cfg: ConfigModelo) -> None:
        super().__init__()
        cfg.validar()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.tamanho_vocab, cfg.d_model)
        self.blocos = nn.ModuleList([BlocoTransformer(cfg) for _ in range(cfg.n_camadas)])
        self.norma_final = RMSNorm(cfg.d_model)
        self.projecao_saida = nn.Linear(cfg.d_model, cfg.tamanho_vocab, bias=False)
        self.projecao_saida.weight = self.embedding.weight
        self.apply(self._inicializar)

    @staticmethod
    def _inicializar(modulo: nn.Module) -> None:
        if isinstance(modulo, nn.Linear):
            nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
        elif isinstance(modulo, nn.Embedding):
            nn.init.normal_(modulo.weight, mean=0.0, std=0.02)

    def forward(
        self,
        ids: torch.Tensor,
        alvos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if ids.size(1) > self.cfg.comprimento_max:
            raise ValueError(
                f"Sequência com {ids.size(1)} tokens excede o limite "
                f"de {self.cfg.comprimento_max}."
            )
        x = self.embedding(ids)
        for bloco in self.blocos:
            x = bloco(x)
        logits = self.projecao_saida(self.norma_final(x))
        perda = None
        if alvos is not None:
            perda = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                alvos.reshape(-1),
            )
        return logits, perda

    @torch.no_grad()
    def gerar(
        self,
        ids: torch.Tensor,
        max_novos_tokens: int,
        temperatura: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_novos_tokens):
            entrada = ids[:, -self.cfg.comprimento_max :]
            logits, _ = self(entrada)
            logits = logits[:, -1, :] / max(temperatura, 1e-5)
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                limite = torch.topk(logits, k).values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < limite, float("-inf"))
            probabilidades = torch.softmax(logits, dim=-1)
            proximo = torch.multinomial(probabilidades, num_samples=1)
            ids = torch.cat((ids, proximo), dim=1)
        return ids

    def contar_parametros(self) -> int:
        return sum(parametro.numel() for parametro in self.parameters())
