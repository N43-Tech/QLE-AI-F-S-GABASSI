from __future__ import annotations

import platform
import sys

import torch


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Sistema: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps"):
        print(f"MPS disponível: {torch.backends.mps.is_available()}")
    print("Ambiente funcional.")


if __name__ == "__main__":
    main()
