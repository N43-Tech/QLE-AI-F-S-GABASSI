from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.model import ConfigModelo, MiniLLM
from src.tokenizer import TokenizadorBytes
from src.train import escolher_dispositivo


NOME_IA = "QLE"
MARCADOR_USUARIO = "USUARIO:"
MARCADOR_IA = f"{NOME_IA}:"


def configurar_utf8() -> None:
    """
    Configura entrada e saída UTF-8, principalmente no Windows.
    Evita textos como 'UsuÃ¡rio' e 'vocÃª'.
    """

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")


def construir_prompt(mensagem: str) -> str:
    """
    Converte uma mensagem comum para o formato usado no treinamento.
    """

    mensagem = mensagem.strip()

    return (
        f"{MARCADOR_USUARIO} {mensagem}\n"
        f"{MARCADOR_IA}"
    )


def cortar_proximo_turno(texto: str) -> str:
    """
    Interrompe a resposta quando o modelo começa um novo turno.
    """

    texto = texto.replace("\r\n", "\n").replace("\r", "\n")

    marcadores = (
        "\nUSUARIO:",
        "\nUsuario:",
        "\nUsuário:",
        "\nQLE:",
        "\nqLE:",
        "\nASSISTENTE:",
        "\nAssistente:",
    )

    posicoes: list[int] = []

    for marcador in marcadores:
        posicao = texto.find(marcador)

        if posicao >= 0:
            posicoes.append(posicao)

    if posicoes:
        texto = texto[: min(posicoes)]

    return texto.strip()


def remover_marcadores_repetidos(texto: str) -> str:
    """
    Remove marcadores que o modelo possa repetir no começo da resposta.
    """

    texto = texto.strip()

    prefixos = (
        "QLE:",
        "qLE:",
        "ASSISTENTE:",
        "Assistente:",
    )

    alterado = True

    while alterado:
        alterado = False

        for prefixo in prefixos:
            if texto.startswith(prefixo):
                texto = texto[len(prefixo):].lstrip()
                alterado = True

    return texto.strip()


def limpar_resposta(texto: str) -> str:
    """
    Aplica as etapas de limpeza na resposta gerada.
    """

    texto = cortar_proximo_turno(texto)
    texto = remover_marcadores_repetidos(texto)

    if not texto:
        return "[A QLE não gerou uma resposta.]"

    return texto


def carregar_modelo(
    caminho_checkpoint: Path,
    dispositivo: torch.device | str,
) -> tuple[MiniLLM, ConfigModelo]:
    """
    Carrega a configuração e os pesos presentes no checkpoint.
    """

    caminho_checkpoint = caminho_checkpoint.resolve()

    if not caminho_checkpoint.exists():
        raise FileNotFoundError(
            "\nCheckpoint não encontrado:\n"
            f"{caminho_checkpoint}\n\n"
            "Coloque o arquivo dentro de checkpoints ou informe "
            "outro caminho usando --checkpoint."
        )

    if not caminho_checkpoint.is_file():
        raise FileNotFoundError(
            f"O caminho não é um arquivo: {caminho_checkpoint}"
        )

    checkpoint = torch.load(
        caminho_checkpoint,
        map_location=dispositivo,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "O checkpoint não possui o formato esperado de dicionário."
        )

    if "config_modelo" not in checkpoint:
        raise KeyError(
            "O checkpoint não possui a chave 'config_modelo'."
        )

    if "estado_modelo" not in checkpoint:
        raise KeyError(
            "O checkpoint não possui a chave 'estado_modelo'."
        )

    configuracao = ConfigModelo(
        **checkpoint["config_modelo"]
    )

    modelo = MiniLLM(configuracao).to(dispositivo)

    modelo.load_state_dict(
        checkpoint["estado_modelo"]
    )

    modelo.eval()

    return modelo, configuracao


def limitar_prompt_ao_contexto(
    prompt_ids: list[int],
    configuracao: ConfigModelo,
) -> list[int]:
    """
    Mantém o final do prompt quando ele ultrapassa o contexto máximo.

    O final é preservado porque contém a pergunta mais recente e o
    marcador 'QLE:'.
    """

    comprimento_max = getattr(
        configuracao,
        "comprimento_max",
        None,
    )

    if comprimento_max is None:
        return prompt_ids

    limite_prompt = max(8, int(comprimento_max) - 1)

    if len(prompt_ids) <= limite_prompt:
        return prompt_ids

    return prompt_ids[-limite_prompt:]


def gerar_resposta(
    modelo: MiniLLM,
    configuracao: ConfigModelo,
    tokenizador: TokenizadorBytes,
    prompt: str,
    dispositivo: torch.device | str,
    quantidade_tokens: int,
    temperatura: float,
    top_k: int,
) -> str:
    """
    Gera somente a continuação produzida pela QLE.
    """

    prompt_ids = tokenizador.codificar(
        prompt,
        adicionar_bos=True,
    )

    prompt_ids = limitar_prompt_ao_contexto(
        prompt_ids,
        configuracao,
    )

    entrada = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=dispositivo,
    )

    with torch.inference_mode():
        saida = modelo.gerar(
            entrada,
            max_novos_tokens=quantidade_tokens,
            temperatura=temperatura,
            top_k=top_k,
        )

    novos_ids = saida[
        0,
        len(prompt_ids):
    ].tolist()

    texto_gerado = tokenizador.decodificar(
        novos_ids
    )

    return limpar_resposta(texto_gerado)


def iniciar_chat(
    modelo: MiniLLM,
    configuracao: ConfigModelo,
    tokenizador: TokenizadorBytes,
    dispositivo: torch.device | str,
    quantidade_tokens: int,
    temperatura: float,
    top_k: int,
) -> None:
    """
    Abre uma conversa interativa pelo terminal.
    """

    print("=" * 55)
    print("QLE iniciada")
    print(f"Dispositivo: {dispositivo}")
    print("Digite 'sair' para encerrar.")
    print("=" * 55)

    while True:
        try:
            mensagem = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nConversa encerrada.")
            break

        if mensagem.lower() in {
            "sair",
            "exit",
            "quit",
            "encerrar",
        }:
            print("Conversa encerrada.")
            break

        if not mensagem:
            continue

        prompt = construir_prompt(mensagem)

        resposta = gerar_resposta(
            modelo=modelo,
            configuracao=configuracao,
            tokenizador=tokenizador,
            prompt=prompt,
            dispositivo=dispositivo,
            quantidade_tokens=quantidade_tokens,
            temperatura=temperatura,
            top_k=top_k,
        )

        print(f"{NOME_IA}: {resposta}")


def criar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa e conversa com a IA QLE."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/qle_base_300_v2.pt"
        ),
        help="Caminho do checkpoint treinado.",
    )

    parser.add_argument(
        "--mensagem",
        type=str,
        default=None,
        help=(
            "Mensagem comum. O programa adiciona automaticamente "
            "os marcadores USUARIO e QLE."
        ),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Prompt completo, usado para testes técnicos. "
            "Exemplo: 'USUARIO: Ola.\\nQLE:'"
        ),
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="Abre o modo de conversa interativa.",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=100,
        help="Quantidade máxima de novos tokens.",
    )

    parser.add_argument(
        "--temperatura",
        type=float,
        default=0.3,
        help="Controla a aleatoriedade da geração.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Quantidade de candidatos considerados por token.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para repetir testes.",
    )

    return parser.parse_args()


def validar_argumentos(args: argparse.Namespace) -> None:
    if args.tokens <= 0:
        raise ValueError(
            "--tokens precisa ser maior que zero."
        )

    if args.temperatura <= 0:
        raise ValueError(
            "--temperatura precisa ser maior que zero."
        )

    if args.top_k <= 0:
        raise ValueError(
            "--top-k precisa ser maior que zero."
        )

    if args.mensagem is not None and args.prompt is not None:
        raise ValueError(
            "Use somente --mensagem ou --prompt, não ambos."
        )


def main() -> None:
    configurar_utf8()

    args = criar_argumentos()
    validar_argumentos(args)

    torch.manual_seed(args.seed)

    dispositivo = escolher_dispositivo()

    modelo, configuracao = carregar_modelo(
        caminho_checkpoint=args.checkpoint,
        dispositivo=dispositivo,
    )

    tokenizador = TokenizadorBytes()

    if args.chat:
        iniciar_chat(
            modelo=modelo,
            configuracao=configuracao,
            tokenizador=tokenizador,
            dispositivo=dispositivo,
            quantidade_tokens=args.tokens,
            temperatura=args.temperatura,
            top_k=args.top_k,
        )

        return

    if args.prompt is not None:
        prompt = args.prompt
    else:
        mensagem = args.mensagem or "Olá."
        prompt = construir_prompt(mensagem)

    resposta = gerar_resposta(
        modelo=modelo,
        configuracao=configuracao,
        tokenizador=tokenizador,
        prompt=prompt,
        dispositivo=dispositivo,
        quantidade_tokens=args.tokens,
        temperatura=args.temperatura,
        top_k=args.top_k,
    )

    print(f"{NOME_IA}: {resposta}")


if __name__ == "__main__":
    main()