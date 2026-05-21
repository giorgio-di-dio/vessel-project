#File per vedere se va il setup e se sono dentro al venv

import torch
import sys
import os


print("-" * 30)
if ".venv" in sys.executable:
    print("STAI USANDO IL VENV! :)")
else:
    print("STAI USANDO IL PYTHON GLOBALE! :(")

print(f"Versione PyTorch: {torch.__version__}")
print(f"GPU disponibile: {torch.cuda.is_available()}")
print("-" * 30)