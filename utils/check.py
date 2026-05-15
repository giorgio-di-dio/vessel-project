import sys
import os

print("-" * 30)
if ".venv" in sys.executable:
    print("STAI USANDO IL VENV! :)")
else:
    print("STAI USANDO IL PYTHON GLOBALE! :(")
print("-" * 30)