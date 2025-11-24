from pathlib import Path
import sys

# Automatically add this directory to Python path when 'compiled' is imported
sys.path.insert(0, str(Path(__file__).parent))

# Optional: expose everything so `from compiled import *` works
from importlib import import_module
import os

__all__ = []
for file in os.listdir(Path(__file__).parent):
    if file.endswith((".pyd", ".so")) and not file.startswith("_"):
        name = file.split(".")[0]
        __all__.append(name)
        try:
            globals()[name] = import_module(name)
        except Exception as e:
            print(f"Warning: Could not import {name}: {e}")
