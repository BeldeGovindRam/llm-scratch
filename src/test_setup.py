import torch
import numpy as np

print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)

print("MPS available:", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

import sys
print(sys.executable)