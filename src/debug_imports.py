# debug_imports.py
import sys

print("--> [1] Testing RDKit Import...")
try:
    from rdkit import Chem
    from rdkit import rdBase
    print("    RDKit OK.")
except Exception as e:
    print(f"    RDKit FAILED: {e}")

print("--> [2] Testing PyTorch Import...")
import torch
print("    PyTorch OK.")

print("--> [3] Testing PyTorch Sparse (Common crash point)...")
try:
    import torch_sparse
    print("    Torch Sparse OK.")
except ImportError:
    print("    Torch Sparse NOT FOUND.")

print("--> [4] Testing PyTorch Scatter...")
try:
    import torch_scatter
    print("    Torch Scatter OK.")
except ImportError:
    print("    Torch Scatter NOT FOUND.")

print("--> [5] Testing PyTorch Lightning...")
import pytorch_lightning
print("    Lightning OK.")

print("--> [6] Testing Local MS_PRED modules...")
try:
    import ms_pred.common as common
    print("    Local modules OK.")
except Exception as e:
    print(f"    Local modules FAILED: {e}")

print("--> SUCCESS: All imports passed.")