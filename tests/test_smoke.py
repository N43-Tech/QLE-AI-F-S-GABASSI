from __future__ import annotations

import unittest

import torch

from src.model import ConfigModelo, MiniLLM
from src.tokenizer import TokenizadorBytes


class TesteMiniLLM(unittest.TestCase):
    def test_tokenizador_reversivel(self) -> None:
        tokenizador = TokenizadorBytes()
        texto = "Olá, automação!"
        ids = tokenizador.codificar(texto)
        self.assertEqual(tokenizador.decodificar(ids), texto)

    def test_forward_e_loss(self) -> None:
        cfg = ConfigModelo(
            d_model=32,
            n_camadas=1,
            n_cabecas=4,
            n_cabecas_kv=2,
            d_ff=64,
            comprimento_max=16,
        )
        modelo = MiniLLM(cfg)
        x = torch.randint(0, cfg.tamanho_vocab, (2, 16))
        y = torch.randint(0, cfg.tamanho_vocab, (2, 16))
        logits, perda = modelo(x, y)
        self.assertEqual(tuple(logits.shape), (2, 16, cfg.tamanho_vocab))
        self.assertIsNotNone(perda)
        assert perda is not None
        self.assertTrue(torch.isfinite(perda))


if __name__ == "__main__":
    unittest.main()
