from tokenizers  import Tokenizer
from tokenizers.models  import BPE
from tokenizers.trainers  import BpeTrainer
from tokenizers. pre_tokenizers import ByteLevel
from tokenizers.processors  import TemplateProcessing
from tokenizers.decoders  import ByteLevel as ByteLevelDecoder
import os

def treinar_tokenizador_minilm (
    arquivos_corpus : list ,
    tamanho_vocab: int  = 32000 ,
    caminho_saida: str  = "./ tokenizador_minilm "
) -> Tokenizer:
    """
    Treina o tokenizador  BPE do MiniLM -PT.

    Parâmetros:
        arquivos_corpus : lista de caminhos  para arquivos .txt
        tamanho_vocab: tamanho do vocabulário final
        caminho_saida: diretório para  salvar o tokenizador
    Retorna:
        tokenizador  treinado
    """
    # 1) Inicializar  modelo BPE com tokens especiais
    tokenizador = Tokenizer(BPE(unk_token=  " <|unk|>"))

    # 2) Pré-tokenização em nível de byte (como GPT -2)
    tokenizador.pre_tokenizer = ByteLevel( add_prefix_space =True)
    tokenizador.decoder = ByteLevelDecoder ()

    # 3) Configurar  treinador
    treinador = BpeTrainer(
        vocab_size=tamanho_vocab ,
        min_frequency =2,            # Ignorar  pares raros
        special_tokens =[
            " <|pad|>",              # ID 0: padding
            " <|unk|>",              # ID 1: desconhecido
            " <|bos|>",              # ID 2: início de sequência
            " <|eos|>",              # ID 3: fim de sequência
        ],
        show_progress=True
    )

    # 4) Treinar nos  arquivos do corpus
    tokenizador.train(arquivos_corpus , treinador)

    # 5) Configurar  processamento: adicionar  BOS/ EOS  automaticamente
    tokenizador. post_processor = TemplateProcessing (
        single= " <|bos|> $A<|eos|>",
        pair=" <|bos|> $A <|eos|> $B:1  <|eos|>:1",
        special_tokens =[
            (" <|bos|>", tokenizador.token_to_id( " <| bos|>")),
            (" <|eos|>", tokenizador.token_to_id( " <| eos|>")),
        ]
    )

    # 6) Salvar
    os.makedirs(caminho_saida , exist_ok=True)
    tokenizador.save(f"{caminho_saida }/ tokenizador.json" )
    print(f"Tokenizador  salvo em: {caminho_saida }/ tokenizador.json")
    print(f"Vocabulário: {tokenizador.get_vocab_size ()} tokens")

    return tokenizador


def analisar_tokenizacao (tokenizador: Tokenizer , textos: list ) -> None:
    """
    Analisa  estatísticas do tokenizador em uma lista de textos.
    """
    total_chars = 0
    total_tokens = 0

    print(f"\n{'Texto ':<50} {'Tokens ':>6} {'Chars ':>6} {'C/T ':>5}")
    print("-" * 70)

    for texto in textos:
        enc = tokenizador.encode(texto)
        n_tokens = len (enc.ids)
        n_chars = len (texto)
        total_chars += n_chars
        total_tokens += n_tokens

        texto_curto = texto [:47] + "..." if len ( texto) > 50 else texto
        print(f"{texto_curto :<50} {n_tokens :>6} { n_chars :>6} {n_chars/n_tokens :>5.2f}")

    print("-" * 70)
    print(f"{'MÉDIA ':<50} {total_tokens :>6} { total_chars :>6} "
        f"{total_chars/total_tokens :>5.2f}" )
