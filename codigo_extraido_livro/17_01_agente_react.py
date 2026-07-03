import anthropic
import json
import re
import subprocess
import requests
from typing  import Any , Callable , Dict , List , Optional

# =======================================================
# Definição de Ferramentas
# =======================================================

FERRAMENTAS_SISTEMA = [
    {
        "name" : " executar_python ",
        "description" : "Executa código  Python e retorna o resultado.",
        "input_schema" : {
            "type" : "object",
            "properties" : {
                "codigo": {
                    "type": "string",
                    "description": "Código  Python a executar"
                }
            },
            "required" : ["codigo"]
        }
    },
    {
        "name" : "buscar_web",
        "description" : "Busca informações na web e retorna os resultados.",
        "input_schema" : {
            "type" : "object",
            "properties" : {
                "consulta": {
                    "type": "string",
                    "description": "Consulta de busca"
                }
            },
            "required" : ["consulta"]
        }
    },
    {
        "name" : "calcular",
        "description" : "Calcula expressões matemá ticas.",
        "input_schema" : {
            "type" : "object",
            "properties" : {
                "expressao": {
                    "type": "string",
                    "description": "Expressão matem ática Python"
                }
            },
            "required" : ["expressao"]
        }
    }
]


def executar_python_seguro (codigo: str) -> str :
    """
    Executa código  Python em um subprocesso  isolado.

    Parâmetros:
        codigo: código  Python a executar
    Retorna:
        saída padrão do código ou mensagem de erro
    """
    try:
        resultado = subprocess.run(
            [ 'python3 ', '-c', codigo],
            capture_output =True , text=True , timeout =10,
            # Ambiente  restrito (sem internet , etc.)
        )
        saida = resultado.stdout.strip ()
        erro = resultado.stderr.strip ()

        if resultado.returncode != 0:
            return f"Erro: {erro}"
        return  saida if saida else "(sem saída)"

    except subprocess. TimeoutExpired:
        return "Erro: timeout (> 10 segundos)"
    except Exception as e:
        return f"Erro ao executar: {str(e)}"


def calcular(expressao:  str) -> str:
    """ Avalia expressão matemática de forma  segura."""
    import math
    contexto_seguro = {
        '__builtins__ ': {},
        'math ': math ,
        'abs': abs , 'round ': round , 'sum': sum ,
        'min': min , 'max': max , 'len': len ,
    }
    try:
        resultado =  eval(expressao , contexto_seguro)
        return str (resultado)
    except Exception as e:
        return f"Erro: {e}"


MAPA_FERRAMENTAS = {
    'executar_python ': executar_python_seguro ,
    'calcular ':        calcular ,
    'buscar_web ':      lambda q: f"[Simulado] Resultados para: {q}",
}


class AgenteReAct:
    """
    Agente LLM com loop  ReAct (Reason + Act).

    O agente  pode:
        - Pensar (gerar  raciocínio interno)
        - Usar  ferramentas (Python , calculadora , busca)
        - Responder ao usuário

    Parâmetros:
        modelo: modelo  Anthropic a usar
        max_iteracoes: máximo de ciclos Razão-Ação
        sistema_prompt : instruções de comportamento
    """

    SISTEMA_PROMPT =  """ Você é um agente especializado em resolver problemas
com o uso de ferramentas. Você deve:
1. Pensar sobre o problema
2. Usar ferramentas  quando necessário para  obter informações ou calcular
3. Sintetizar os resultados em uma  resposta  clara e precisa

Sempre explique  seu raciocínio antes de usar uma ferramenta."""

    def __init__(self ,
                modelo: str = "claude -3-5-haiku -20241022",
                max_iteracoes: int = 10):
        self.cliente = anthropic.Anthropic ()
        self.modelo = modelo
        self.max_iteracoes = max_iteracoes
        self.historico: List[ dict] = []
        self. n_chamadas_ferramenta = 0

    def _executar_ferramenta (self , nome: str , params: dict) -> str:
        """ Executa uma ferramenta e retorna o resultado."""
        ferramenta_fn = MAPA_FERRAMENTAS .get(nome)
        if ferramenta_fn  is None:
            return f"Ferramenta '{nome}' não encontrada."

        self. n_chamadas_ferramenta += 1

        # Extrair  primeiro parâmetro
        param = list (params.values ())[0] if params else ""
        resultado = ferramenta_fn(param)

        print(f"  [Ferramenta: {nome }] {param [:50]}...")
        print(f"  [Resultado] {str(resultado) [:80]}...")
        return str (resultado)

    def executar(self , tarefa: str) -> str :
        """
        Executa o agente em uma tarefa.

        Parâmetros:
            tarefa: descrição da tarefa  para o agente
        Retorna:
            resposta final do agente
        """
        print(f"\nAgente iniciado: {tarefa [:60]}...\n")

        # Mensagem  inicial do usuário
        self.historico = [{ "role": "user" , "content ": tarefa }]

        for  iteracao in range(self.max_iteracoes):
            # Chamar o modelo
            resposta = self.cliente.messages.create (
                model=self.modelo ,
                max_tokens =2048 ,
                system=self.SISTEMA_PROMPT ,
                tools=FERRAMENTAS_SISTEMA ,
                messages=self.historico
            )

            # Verificar se o agente  terminou
            if resposta.stop_reason ==  "end_turn" :
                # Extrair texto da resposta  final
                texto_final = ""
                for bloco in resposta.content:
                    if hasattr(bloco , 'text '):
                        texto_final = bloco.text
                        break

                print(f"\n[Agente finalizado após { iteracao +1} iteração(ões)]")
                print(f"[Ferramentas usadas: {self.n_chamadas_ferramenta }]")
                return texto_final

            # Processar  uso de ferramentas
            self.historico.append ({
                "role": "assistant",
                "content": resposta.content
            })

            resultados_ferramentas = []

            for bloco  in resposta.content:
                if bloco.type == "tool_use" :
                    resultado = self._executar_ferramenta (
                        bloco.name , bloco.input
                    )
                    resultados_ferramentas .append ({
                        "type": "tool_result" ,
                        "tool_use_id": bloco. id ,
                        "content": resultado
                    })

            if  resultados_ferramentas :
                self.historico.append ({
                    "role": "user",
                    "content": resultados_ferramentas
                })

        return "Limite de iterações atingido."


# Demonstração
agente = AgenteReAct ()
print("Agente  ReAct inicializado!")
print("\nExemplo de tarefa:" )
print("  'Calcule a soma dos 100  primeiros números primos '")
print("\nO agente  iria:")
print("  1. Pensar  sobre a abordagem")
print("  2. Chamar  executar_python com código de crivo de Eratóstenes" )
print("  3. Sintetizar o resultado" )
