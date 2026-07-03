from fastapi  import FastAPI
from fastapi.responses  import StreamingResponse
from pydantic  import BaseModel
import torch
import asyncio
import time
import json

app = FastAPI(title= "MiniLM -PT ServingAPI" )

# Carregar modelo (global , compartilhado  entre requests)
MODELO = None
TOKENIZADOR = None

class RequestInferencia (BaseModel):
    prompt: str
    max_tokens: int = 200
    temperatura: float  = 0.8
    top_p: float = 0.9
    stream: bool = True


async def gerar_tokens_stream (prompt: str ,
                                max_tokens: int ,
                                temperatura: float ,
                                top_p: float ):
    """
    Gerador assíncrono que produz  tokens um a um.
    Permite  streaming HTTP via Server -Sent Events.
    """
    # Tokenizar  prompt
    ids_prompt = torch.tensor(
        [TOKENIZADOR.tokenizar(prompt)], dtype= torch.long
    )

    ids = ids_prompt
    cache_kv = None
    t_inicio = time.time ()
    n_tokens = 0

    for i in range (max_tokens):
        if cache_kv is  None:
            entrada = ids
        else:
            entrada = ids[:, -1:]

        with  torch.no_grad ():
            logits , cache_kv = MODELO(
                entrada , cache_kvs=cache_kv , retornar_cache =True
            )

        # Sampling
        probs = torch.softmax(logits [:, -1, :] / temperatura , dim=-1)
        probs_sort , indices_sort = torch.sort(probs, descending=True)
        probs_cum = torch.cumsum(probs_sort , dim =-1)
        probs_sort[probs_cum - probs_sort > top_p] = 0
        probs_sort /= probs_sort. sum()

        token_idx = torch.multinomial(probs_sort , 1)
        token = torch.gather(indices_sort , -1, token_idx)
        ids = torch.cat([ids , token], dim=-1)

        n_tokens += 1

        # Decodificar  token
        texto_token = TOKENIZADOR.decodificar ([ token.item ()])

        # SSE payload
        dados = {
            'token ': texto_token ,
            'token_id ': token.item (),
            'n_token ': n_tokens ,
            'tokens_por_segundo ': n_tokens / (time.time () - t_inicio),
            'finalizado ': (token.item () == 3)
        }
        yield f"data: {json.dumps(dados)}\n\n"

        if token.item () == 3:
            break

        # Simular  processamento assíncrono
        await  asyncio.sleep (0)


@app.post("/gerar" )
async def endpoint_gerar (request: RequestInferencia):
    """
    Endpoint de geração com  suporte a streaming.

    Exemplo de uso:
        curl -X POST http :// localhost :8000/ gerar \\
                -H "Content -Type: application/json" \\
                -d '{" prompt ": "A inteligência artificial", "stream ": true}'
    """
    if request.stream:
        return  StreamingResponse (
            gerar_tokens_stream (
                request.prompt ,
                request.max_tokens ,
                request.temperatura ,
                request.top_p
            ),
            media_type="text/event -stream"
        )
    else:
        # Resposta  completa (sem streaming)
        tokens = []
        async for evento  in gerar_tokens_stream (
            request.prompt , request.max_tokens ,
            request.temperatura , request.top_p
        ):
            dados = json.loads(evento.replace( "data: ", ""))
            tokens.append(dados['token '])

        return { "texto": "".join(tokens), "n_tokens ": len(tokens)}


@app.get("/saude" )
async def saude ():
    """ Health check endpoint."""
    return {
        "status" : "ok",
        "modelo" : "MiniLM -PT",
        "dispositivo" : str(next(MODELO.parameters ()).device) if MODELO else "N/A"
    }

# Para executar: uvicorn  cap14:app --host  0.0.0.0 --port 8000
