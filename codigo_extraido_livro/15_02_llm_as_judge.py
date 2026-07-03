import anthropic
import json
from typing  import List , Tuple

RUBRICA_JUIZ = """
Você é um juiz  imparcial avaliando a qualidade de respostas de IA.
Avalie a resposta  abaixo em uma escala de 1 a 10 considerando:
- Precisão factual (0-3 pontos)
- Completude (0-3 pontos)
- Clareza e estrutura (0-2 pontos)
- Utilidade (0-2 pontos)

Pergunta: {pergunta}
Resposta: {resposta}

Retorne um JSON com: {{" pontuacao ": X, " justificativa ": "..."}}
"""

def avaliar_com_llm_juiz (
    pergunta: str ,
    resposta: str ,
    modelo_juiz: str  = "claude -3-5-haiku -20241022"
) -> Tuple[float , str]:
    """
    Usa um LLM para  avaliar a qualidade de uma resposta.

    Parâmetros:
        pergunta: a pergunta  original
        resposta: resposta do modelo a ser   avaliado
        modelo_juiz: modelo a usar como juiz
    Retorna:
        (pontuacao , justificativa)
    """
    cliente = anthropic.Anthropic ()

    prompt = RUBRICA_JUIZ. format(
        pergunta=pergunta ,
        resposta=resposta
    )

    mensagem = cliente.messages.create(
        model=modelo_juiz ,
        max_tokens =512,
        messages =[{"role": "user", "content" : prompt }]
    )

    try:
        resultado = json.loads(mensagem.content [0].text)
        return  float(resultado['pontuacao ']), resultado['justificativa ']
    except (json.JSONDecodeError , KeyError):
        return 0.0,  "Erro ao parsear resposta do juiz"


def avaliar_suite(
    pares_qa: List[Tuple[ str , str]],
    modelo_juiz: str  = "claude -3-5-haiku -20241022"
) -> dict:
    """
    Avalia uma suíte de pares  pergunta -resposta.

    Parâmetros:
        pares_qa: lista de (pergunta , resposta)
    Retorna:
        estatísticas da avaliação
    """
    pontuacoes = []

    for i, (pergunta , resposta) in enumerate ( pares_qa):
        pontuacao , justificativa = avaliar_com_llm_juiz (
            pergunta , resposta , modelo_juiz
        )
        pontuacoes.append(pontuacao)
        print(f"Q{i+1}: {pontuacao :.1f}/10 | { justificativa [:60]}...")

    return {
        'media ':   sum(pontuacoes) / len(pontuacoes),
        'minimo ':  min(pontuacoes),
        'maximo ':  max(pontuacoes),
        'n_avaliacoes ': len(pontuacoes)
    }


# Exemplo de uso com  respostas do MiniLM -PT
exemplos = [
    ("O que é um Transformer?" ,
    "Um Transformer é uma  arquitetura de rede neural baseada em atenção."),
    ("Como funciona o mecanismo de atenção?"  ,
    "O mecanismo de atenção calcula a relevância de cada token em relação aos outros." ),
]

print("=== Avaliação LLM -as -Judge ===")
for pergunta , resposta in exemplos:
    print(f"\nPergunta: {pergunta}" )
    print(f"Resposta: {resposta}" )
    print(f"(Avaliaria  com modelo Anthropic se API estivesse disponível)")
