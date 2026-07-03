$ErrorActionPreference = "Stop"
& .\.venv\Scripts\python.exe -m src.train --passos 10
& .\.venv\Scripts\python.exe -m src.generate --prompt "A inteligência artificial" --tokens 40
