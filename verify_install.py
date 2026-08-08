import platform
import torch
import torchvision
import lightly
import numpy as np
import sklearn

print("=" * 60)
print("SSL Comparison Framework - Environment Check")
print("=" * 60)

print(f"Python       : {platform.python_version()}")
print(f"PyTorch      : {torch.__version__}")
print(f"Torchvision  : {torchvision.__version__}")
print(f"Lightly      : {lightly.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Scikit-Learn : {sklearn.__version__}")

print("\nDevice Information")
print("-" * 60)

print(f"MPS Built     : {torch.backends.mps.is_built()}")
print(f"MPS Available : {torch.backends.mps.is_available()}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using Device  : {device}")

print("\nEnvironment successfully configured!")