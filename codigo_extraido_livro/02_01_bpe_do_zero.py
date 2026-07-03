from collections  import Counter , defaultdict
from typing  import Dict , List , Tuple , Optional
import re

class TreinadorBPE:
    """
    Implementação completa do Byte -Pair  Encoding ( BPE).

    Referência:
        Sennrich et al. (2016) - Neural   Machine Translation of Rare Words
        with  Subword Units. ACL 2016.
    """

    def __init__(self , tamanho_vocab: int = 32000):
        self.tamanho_vocab = tamanho_vocab
        self.vocab: Dict[ str , int] = {}
        self.fusoes: List[Tuple[ str , str]] = []
        self.vocab_inverso: Dict[ int , str] = {}

    def _pre_tokenizar (self , texto: str) -> List[ str ]:
        """
        Pré-tokenização: divide por espaços com marcador especial G_dot.
        G_dot = marcador de início de palavra.
        """
        # Padrão GPT -2: divide por espaço, preservando pontuação
        padrao = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+"""
        palavras = re.findall(padrao , texto)
        # Adicionar  marcador de espaço como  prefixo
        resultado = []
        for  palavra in palavras:
            if palavra.startswith( ' '):
                resultado.append('\u0120 ' + palavra [1:])
            else :
                resultado.append(palavra)
        return  resultado

    def _contar_pares(self ,
                        vocab_freq: Dict[Tuple[ str,...] , int]
                        ) -> Dict[Tuple[str ,str], int]:
        """
        Conta a frequência de todos os pares adjacentes no vocabulário.

        Parâmetros:
            vocab_freq: dicionário (sequência_chars -> frequência)
        Retorna:
            contagem de pares (char_a , char_b) -> frequência
        """
        pares: Dict[Tuple[ str ,str], int] = defaultdict(int)

        for sequencia , freq in vocab_freq.items ():
            simbolos = list(sequencia)
            for i in range (len(simbolos) - 1):
                pares [( simbolos[i], simbolos[i+1])] += freq

        return  pares

    def _aplicar_fusao (self ,
                        vocab_freq: Dict[Tuple[ str,...] , int],
                        par: Tuple[str , str]
                        ) -> Dict[Tuple[str ,...] , int]:
        """
        Aplica uma fusão a todas as palavras do vocabulário.
        Substitui  todas as ocorrências do par (a, b) por 'ab '.
        """
        novo_vocab: Dict[Tuple[ str ,...] , int] = {}
        a, b = par
        novo_token = a + b

        for sequencia , freq in vocab_freq.items ():
            nova_seq = []
            i = 0
            simbolos = list(sequencia)

            while i < len(simbolos):
                if (i < len(simbolos) - 1 and
                    simbolos[i] == a and  simbolos[i +1] == b):
                    nova_seq.append(novo_token)
                    i += 2
                else:
                    nova_seq.append(simbolos[i])
                    i += 1

            novo_vocab[tuple(nova_seq)] = freq

        return  novo_vocab

    def treinar(self , corpus: str ,
                verbose: bool = True ,
                intervalo_log: int = 1000)  -> None:
        """
        Treina o tokenizador  BPE em um corpus de texto.

        Parâmetros:
            corpus: texto bruto de treinamento
            verbose: se True , imprimeprogresso
            intervalo_log: frequência de logs
        """
        # 1) Pré-tokenizar e contar  frequências de palavras
        palavras = self. _pre_tokenizar(corpus)
        contagem_palavras : Counter = Counter( palavras)

        # 2) Inicializar: cada  palavra  como sequê ncia de chars
        vocab_freq: Dict[Tuple[ str ,...] , int] = {
            tuple(palavra): freq
            for palavra , freq in contagem_palavras .items ()
        }

        # 3) Vocabulário base = todos os chars ú nicos
        chars_unicos =  set()
        for seq in vocab_freq:
            chars_unicos.update(seq)

        # Tokens  especiais
        tokens_especiais = [ '<|pad|>', '<|unk|>' , ' <|bos|>', '<|eos|>']
        self.vocab = {tok: i  for i, tok in enumerate( tokens_especiais )}
        for ch in sorted (chars_unicos):
            if ch not in  self.vocab:
                self.vocab[ch] = len(self.vocab)

        n_fusoes = self.tamanho_vocab -  len (self.vocab)

        if verbose:
            print(f"Vocabulário inicial: {len(self.vocab)} tokens")
            print(f"Fusões necessárias: {n_fusoes}")
            print(f"Palavras únicas no corpus: {len (vocab_freq)}")

        # 4) Loop BPE
        for i in range (n_fusoes):
            pares = self._contar_pares(vocab_freq)

            if not pares:
                break

            melhor_par = max(pares , key=lambda p: pares[p])
            freq_melhor = pares[melhor_par]

            if freq_melhor < 2:
                print(f"Parando: par mais  frequente tem freq ={ freq_melhor}")
                break

            vocab_freq = self. _aplicar_fusao( vocab_freq , melhor_par)

            novo_token = melhor_par [0] + melhor_par [1]
            self.fusoes.append(melhor_par)
            self.vocab[novo_token] =  len(self.vocab)

            if verbose  and (i + 1) % intervalo_log == 0:
                print(f"Fusão {i+1:6d}/{ n_fusoes} | "
                        f" '{melhor_par [0]}' + '{ melhor_par [1]}' "
                        f"-> '{novo_token}' (freq ={ freq_melhor :,})")

        self.vocab_inverso = {v: k  for k, v in  self.vocab.items ()}

        if verbose:
            print(f"\nTreinamento concluído!" )
            print(f"Vocabulário final: {len(self.vocab)} tokens")

    def tokenizar(self , texto: str) -> List[ int]:
        """
        Tokeniza  texto usando as fusões aprendidas.

        Parâmetros:
            texto: texto a tokenizar
        Retorna:
            lista de IDs de tokens
        """
        palavras = self. _pre_tokenizar(texto)
        ids = []

        for  palavra in palavras:
            sequencia = list(palavra)

            # Aplicar fusões na ordem  aprendida
            for (a, b)  in self.fusoes:
                nova_seq = []
                i = 0
                while i < len(sequencia):
                    if (i < len(sequencia) - 1  and
                        sequencia[i] == a and sequencia[i+1] == b):
                        nova_seq.append(a + b)
                        i += 2
                    else:
                        nova_seq.append(sequencia[i])
                        i += 1
                sequencia = nova_seq

            for token  in sequencia:
                ids.append(self.vocab.get(token , self.vocab['<|unk|>' ]))

        return ids

    def decodificar(self , ids: List[int]) -> str :
        """
        Converte  IDs de volta para texto.
        """
        tokens = [self.vocab_inverso.get(i,  '<|unk |>') for i in ids]
        texto = '' .join(tokens)
        return  texto.replace('\u0120 ', ' ').strip ()


# Demonstração com corpus  pequeno
corpus_demo = """
O aprendizado de máquina  transformou a inteligência artificial  moderna.
Modelos como o Gemini  processam bilhões de tokens durante o treinamento.
A arquitetura  Transformer revolucionou o processamento de linguagem  natural.
Tokenizadores  eficientes são fundamentais  para o desempenho de LLMs.
"""

bpe = TreinadorBPE(tamanho_vocab =500)
bpe.treinar(corpus_demo , verbose=True , intervalo_log =50)

# Tokenizar uma frase
frase = "Modelos de linguagem  aprendem  padrões"
ids = bpe.tokenizar(frase)
tokens = [bpe.vocab_inverso.get(i,  '?') for i  in ids]

print(f"\nFrase: '{frase}'" )
print(f"Tokens: {tokens}" )
print(f"IDs:     {ids}")
print(f"Reconstrução: '{bpe.decodificar(ids)}'"  )
print(f"Taxa de compressão: {len(frase)/len(ids):.2 f} chars/token" )
