# Validação realizada

- Todos os arquivos Python em `src/` e `codigo_extraido_livro/` passaram por compilação sintática com `compileall`.
- O tokenizador passou por teste de ida e volta UTF-8.
- O modelo passou por teste de forward, cálculo de cross-entropy e geração.
- O loop de treinamento foi executado em CPU e produziu um checkpoint válido.

Execute localmente:

```powershell
python -m unittest discover -s tests -v
python -m src.train --passos 10
```
