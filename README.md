# IA do zero no VS Code - Mini LLM em PyTorch

Projeto educacional organizado a partir dos códigos e conceitos do livro enviado.

## Início rápido no Windows

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
python -m src.train --passos 10
python -m src.generate --prompt "A inteligência artificial" --tokens 40
```

Leia `GUIA_PASSO_A_PASSO.md` antes de aumentar o modelo.

## Estrutura

- `src/`: implementação executável e testada do mini LLM;
- `codigo_extraido_livro/`: todos os blocos numerados de código extraídos do PDF;
- `config_tiny.json`: configuração indicada para o primeiro teste;
- `config_livro_aproximada.json`: configuração arquitetural maior, não indicada para CPU;
- `data/corpus.txt`: corpus demonstrativo;
- `.vscode/`: tarefas e configurações de depuração.
