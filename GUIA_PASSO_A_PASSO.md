# Guia prático: do PDF ao primeiro mini LLM no VS Code

## 1. O que este projeto realmente entrega

Este projeto implementa um **modelo de linguagem causal pequeno**, treinado do zero em PyTorch. Ele usa os componentes centrais apresentados no livro:

1. tokenização;
2. embeddings e compartilhamento de pesos;
3. atenção causal com GQA;
4. RoPE;
5. RMSNorm;
6. SwiGLU;
7. blocos Transformer em arquitetura Pre-Norm;
8. entropia cruzada, AdamW, warmup e decaimento cosseno;
9. geração autoregressiva.

Não é um Gemini ou Claude. Esses sistemas exigem enormes conjuntos de dados, muitas GPUs, engenharia distribuída, alinhamento e infraestrutura de produção. O objetivo aqui é construir a mesma **categoria arquitetural**, em escala educacional.

## 2. Abrir no VS Code no Windows

1. Extraia o ZIP em uma pasta simples, por exemplo `C:\IA_do_zero_VSCode`.
2. Abra o VS Code.
3. Use **Arquivo > Abrir Pasta** e selecione a pasta extraída.
4. Instale as extensões oficiais **Python** e **Python Debugger** da Microsoft.
5. Abra o terminal integrado com `Ctrl + J`.

## 3. Criar o ambiente virtual

No PowerShell do VS Code:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

Depois pressione `Ctrl + Shift + P`, procure **Python: Select Interpreter** e selecione:

```text
.venv\Scripts\python.exe
```

Teste:

```powershell
python -m src.check_environment
```

## 4. Executar um teste curto

```powershell
python -m src.train --passos 10
```

Esse comando valida o fluxo completo: corpus -> tokens -> batches -> forward -> loss -> backward -> AdamW -> checkpoint.

## 5. Treinar a configuração pequena

```powershell
python -m src.train --config config_tiny.json
```

O checkpoint será criado em:

```text
checkpoints/modelo.pt
```

Com o corpus de demonstração, o modelo apenas memoriza padrões locais. Para aprender português minimamente útil, substitua `data/corpus.txt` por um corpus limpo muito maior e aumente os passos gradualmente.

## 6. Gerar texto

```powershell
python -m src.generate --prompt "A inteligência artificial" --tokens 120
```

Para reduzir aleatoriedade:

```powershell
python -m src.generate --prompt "Automação industrial" --tokens 120 --temperatura 0.5 --top-k 20
```

## 7. Ordem correta para estudar os arquivos

### Etapa A - Fundamentos

- `src/tokenizer.py`: transforma texto UTF-8 em IDs.
- `codigo_extraido_livro/01_01_weight_tying.py`: embedding e unembedding compartilham a mesma matriz.
- `codigo_extraido_livro/01_02_contagem_parametros.py`: mede o tamanho do modelo.

### Etapa B - Núcleo do Transformer

- `src/model.py`, classe `RMSNorm`;
- `src/model.py`, classe `RoPE`;
- `src/model.py`, classe `AtencaoCausalGQA`;
- `src/model.py`, classe `SwiGLU`;
- `src/model.py`, classe `BlocoTransformer`;
- `src/model.py`, classe `MiniLLM`.

Compare com as listagens 3.1, 4.1, 4.2 e 4.3 extraídas do livro.

### Etapa C - Treinamento

- `src/dataset.py`: cria pares entrada/alvo deslocados por um token;
- `src/train.py`: calcula cross-entropy, backpropagation, clipping, AdamW e scheduler;
- `config_tiny.json`: hiperparâmetros editáveis.

Compare com as listagens 5.1 e A.1.

### Etapa D - Inferência

- `src/generate.py` carrega o checkpoint;
- `MiniLLM.gerar` executa amostragem autoregressiva com temperatura e top-k.

Compare com a listagem 5.2.

### Etapa E - Tópicos avançados

Somente depois que o projeto básico estiver claro, avance para:

- MoE: capítulo 6;
- atenção longa e Mamba: capítulo 7;
- multimodalidade: capítulo 8;
- DPO: capítulo 10;
- LoRA/QLoRA: capítulo 11;
- FSDP: capítulo 12;
- serving FastAPI: capítulo 14;
- agentes e interpretabilidade: capítulos 17 e 18.

Misturar tudo desde o início é uma estratégia ruim. Primeiro faça o modelo causal simples treinar e gerar; depois substitua um componente por vez.

## 8. Como trocar o corpus

Edite ou substitua:

```text
data/corpus.txt
```

Regras mínimas:

- use UTF-8;
- remova HTML, menus repetidos, lixo e dados pessoais;
- evite textos sem autorização de uso;
- mantenha diversidade de assuntos e estilos;
- não treine com senhas, chaves de API ou dados privados.

## 9. Como aumentar o modelo sem travar o computador

Aumente um fator por vez no `config_tiny.json`:

1. `d_model`: 128 -> 192 -> 256;
2. `n_camadas`: 4 -> 6 -> 8;
3. `comprimento_max`: 128 -> 256 -> 512;
4. `batch_size`: reduza se faltar memória.

A configuração `config_livro_aproximada.json` é pesada. Ela não é apropriada para um primeiro teste em CPU e pode exceder a memória de GPUs comuns durante treinamento, porque os parâmetros são apenas uma parte do consumo; gradientes, estados do AdamW e ativações também ocupam memória.

## 10. Erros comuns

### `ModuleNotFoundError: No module named 'torch'`

O interpretador selecionado não é o `.venv` ou as dependências não foram instaladas.

### PowerShell bloqueou o script

Execute:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### Treinamento muito lento

Use `--passos 10`, reduza `d_model`, `n_camadas`, `comprimento_max` e `batch_size`. Em CPU, treinamento de Transformer é lento.

### Texto gerado sem sentido

Isso é esperado com corpus pequeno e poucos passos. Arquitetura correta não compensa dados insuficientes.
