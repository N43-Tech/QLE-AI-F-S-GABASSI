import anthropic
from dataclasses  import dataclass
from typing  import List , Optional
import heapq

@dataclass
class Pensamento:
    """Nó na árvore de raciocínio."""
    conteudo: str
    valor: float = 0.0
    profundidade: int  = 0
    pai: Optional[ 'Pensamento '] = None
    filhos: List[ 'Pensamento '] = None

    def __post_init__(self):
        self.filhos = []  if self.filhos is None else self.filhos

    def __lt__(self , outro):
        return  self.valor > outro.valor   # Max -heap


class BuscaArvoredePensamento :
    """
    Tree -of -Thoughtcom busca por largura e auto - avaliação.

    Referência:
        Yao et al. (2023). Tree of Thoughts: Deliberate Problem Solving
        with  Large Language Models. NeurIPS  2023.

    Parâmetros:
        n_candidatos: pensamentos  por nível
        profundidade_max : profundidade máxima da á rvore
        n_melhores: manter os n_melhores em cada ní vel (BFS beam)
    """

    def __init__(self ,
                n_candidatos: int = 3,
                profundidade_max : int = 4,
                n_melhores: int = 3):
        self.n_candidatos = n_candidatos
        self. profundidade_max = profundidade_max
        self.n_melhores = n_melhores
        self.cliente = anthropic.Anthropic ()

    def _gerar_pensamentos (self ,
                                problema: str ,
                                historico: str ,
                                n: int) -> List[ str]:
        """ Gera n pensamentos candidatos  para o pró ximo passo."""
        prompt = f""" Problema: {problema}

Raciocínio até agora:
{historico}

Gere {n} próximos  passos de raciocínio  DISTINTOS ( um por linha , numerados):
Lembre -se: cada passo deve ser conciso e avançar a solução."""

        resposta = self.cliente.messages.create(
            model="claude -3-5-haiku -20241022",
            max_tokens =512,
            messages =[{"role": "user", "content": prompt }]
        )

        linhas = resposta.content [0]. text.strip ().split('\n')
        pensamentos = []
        for linha in  linhas:
            linha = linha.strip ()
            if linha  and not linha.startswith( '#'):
                # Remover numeração
                if '. ' in linha [:5]:
                    linha = linha.split( '. ', 1)[1]
                pensamentos.append(linha)

        return  pensamentos [:n]

    def _avaliar_pensamento (self ,
                                problema: str ,
                                raciocinio: str ) -> float :
        """ Avalia a qualidade/progresso de um caminho de raciocínio."""
        prompt = f""" Problema: {problema}

Raciocínio tentado:
{raciocinio}

Avalie o progresso  deste raciocínio em uma escala de 0.0 a 1.0:
- 0.0: completamente  errado ou sem progresso
- 0.5: algum  progresso , masincompleto
- 1.0: solução correta ou quase  completa

Responda APENAS com um número  decimal entre 0.0e1.0. """

        resposta = self.cliente.messages.create(
            model="claude -3-5-haiku -20241022",
            max_tokens =16,
            messages =[{"role": "user", "content": prompt }]
        )

        try:
            valor = float(resposta.content [0]. text.strip ())
            return max (0.0, min (1.0, valor))
        except  ValueError:
            return 0.5

    def resolver(self , problema: str) -> dict :
        """
        Resolve um problema  usando busca em árvore de pensamentos.

        Parâmetros:
            problema: descrição do problema
        Retorna:
            dicionário com solução e estatísticas
        """
        raiz = Pensamento(conteudo= "[Início]" , valor =1.0)
        fila = [raiz]   # BFS beam

        melhor_solucao = None
        melhor_valor =  -1.0
        total_nós = 1

        for nivel in range (self. profundidade_max ):
            candidatos = []

            for nó in  fila:
                # Construir historico do caminho at é o nó
                caminho = []
                atual = nó
                while atual.pai is not None:
                    caminho.insert (0, atual.conteudo)
                    atual = atual.pai
                historico = '\n'.join(caminho)  if caminho else "[Começar]"

                # Gerar pensamentos filhos
                pensamentos = self._gerar_pensamentos (
                    problema , historico , self.n_candidatos
                )

                for pensamento in pensamentos:
                    novo_historico = historico +  '\n' + pensamento
                    valor = self._avaliar_pensamento ( problema , novo_historico )

                    filho = Pensamento(
                        conteudo=pensamento ,
                        valor=valor ,
                        profundidade=nivel + 1,
                        pai=nó
                    )
                    nó.filhos.append(filho)
                    candidatos.append(filho)
                    total_nós += 1

                    if valor > melhor_valor:
                        melhor_valor = valor
                        melhor_solucao = filho

            # Selecionar os n_melhores   para o pró ximo nível
            fila = sorted (candidatos , key=lambda n: n.valor , reverse=True)
            fila = fila [: self.n_melhores]

            print(f"Nível {nivel +1}: {len( candidatos)} nós, "
                    f"melhor valor = {melhor_valor :.3 f}")

            if melhor_valor  >= 0.95:
                print("Solução de alta  confiança encontrada!")
                break

        # Reconstruir  caminho da melhor  solução
        caminho_final = []
        atual = melhor_solucao
        while  atual.pai is not None:
            caminho_final.insert (0, atual.conteudo)
            atual = atual.pai

        return {
            'solucao ': '\n'.join(caminho_final),
            'confianca ': melhor_valor ,
            'nos_explorados ': total_nós,
            'profundidade ': melhor_solucao .profundidade if melhor_solucao  else 0
        }


# Demonstração
problema_demo = """João tem 3 vezes  mais maçãs que Maria.
Juntos eles têm 48 maçãs. Quantas maçãs cada um tem ?"""

print(f"Problema: {problema_demo}" )
print("(Tree -of -Thoughtexigiria conexão com API Anthropic)" )
print("\nEstrutura do algoritmo:" )
print("  1. Gerar 3 pensamentos  candidatos" )
print("  2. Avaliar  cada um (0.0 -1.0)")
print("  3. Expandir os 3 melhores" )
print("  4. Repetir até profundidade 4 ou valor   >= 0.95")
