import nbformat as nbf
from pathlib import Path

# Create a new notebook object
nb = nbf.v4.new_notebook()

# Cell 1: Setup & Model
cell_1 = """import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class MS3ReRanker(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def print_tensor_state(name, tensor_or_list):
    if isinstance(tensor_or_list, torch.Tensor):
        print(f"✅ {name:<25} | Type: Tensor | Shape: {str(list(tensor_or_list.shape)):<15} | Dtype: {tensor_or_list.dtype}")
    elif isinstance(tensor_or_list, list):
        print(f"✅ {name:<25} | Type: List   | Length: {str(len(tensor_or_list)):<14} | Inner: {type(tensor_or_list[0]).__name__}")
    elif isinstance(tensor_or_list, np.ndarray):
        print(f"✅ {name:<25} | Type: NumPy  | Shape: {str(list(tensor_or_list.shape)):<15} | Dtype: {tensor_or_list.dtype}")"""

# Cell 2: Mocking the DataLoader & Actor
cell_2 = """BATCH_SIZE = 64
TOP_K = 50
MAX_MS3_PEAKS = 120
FEATURE_DIM = 13

print("--- 1. DATALOADER OUTPUT ---")
mock_smiles_list = ["CC(=O)OC1=CC=CC=C1C(=O)O"] * BATCH_SIZE
mock_ces = torch.full((BATCH_SIZE,), 30.0, dtype=torch.float32)
mock_ms2_mz = torch.rand((BATCH_SIZE,), dtype=torch.float32) * 500 + 100
mock_ms3_mz = torch.rand((BATCH_SIZE, MAX_MS3_PEAKS), dtype=torch.float32) * 500
mock_ms3_int = torch.rand((BATCH_SIZE, MAX_MS3_PEAKS), dtype=torch.float32)

print_tensor_state("smiles_list", mock_smiles_list)
print_tensor_state("batched_ces", mock_ces)
print_tensor_state("batched_ms2_mz", mock_ms2_mz)
print_tensor_state("padded_ms3_mz", mock_ms3_mz)
print_tensor_state("padded_ms3_int", mock_ms3_int)

print("\\n--- 2. ACTOR (ICEBERG) GENERATION ---")
mock_candidate_smiles = [["C1=CC=CC=C1"] * TOP_K for _ in range(BATCH_SIZE)]
print_tensor_state("candidate_smiles", mock_candidate_smiles)

TOTAL_FRAGMENTS = BATCH_SIZE * TOP_K
mock_features = torch.randn((TOTAL_FRAGMENTS, FEATURE_DIM), dtype=torch.float32)
print_tensor_state("Critic Input Features", mock_features)"""

# Cell 3: Oracle Scoring & Critic Prediction
cell_3 = """print("--- 3. ORACLE (MAGMa) SCORING ---")
mock_magma_rewards = np.random.uniform(0.0, 1.0, size=(TOTAL_FRAGMENTS,))
true_rewards_tensor = torch.tensor(mock_magma_rewards, dtype=torch.float32).unsqueeze(-1)
print_tensor_state("Ground Truth Rewards", true_rewards_tensor)

print("\\n--- 4. CRITIC (Re-Ranker) PREDICTION ---")
model = MS3ReRanker(input_dim=FEATURE_DIM)
predicted_scores = model(mock_features)
print_tensor_state("Predicted Scores", predicted_scores)

assert predicted_scores.shape == true_rewards_tensor.shape, "CRITICAL ERROR: Shape mismatch!"
print("\\n✅ SHAPE MATCH VERIFIED: Predictions and Rewards align perfectly.")"""

# Cell 4: Backprop & Gradients
cell_4 = """print("--- 5. LOSS & BACKPROPAGATION ---")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss = criterion(predicted_scores, true_rewards_tensor)
print(f"Initial MSE Loss: {loss.item():.4f}")

optimizer.zero_grad()
loss.backward()

first_layer_grad = model.net[0].weight.grad
print_tensor_state("Layer 1 Gradients", first_layer_grad)

if first_layer_grad is not None and not torch.all(first_layer_grad == 0):
    print("\\n🚀 BACKPROPAGATION SUCCESSFUL! Gradients are flowing through the Critic.")
else:
    print("\\n❌ GRADIENT FAILURE! The computation graph broke.")

optimizer.step()"""

# Add cells to notebook
nb.cells.extend([
    nbf.v4.new_code_cell(cell_1),
    nbf.v4.new_code_cell(cell_2),
    nbf.v4.new_code_cell(cell_3),
    nbf.v4.new_code_cell(cell_4)
])

# Write the notebook to disk
output_path = Path("fiar_pipeline/notebooks/01_architecture_tracer.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✅ Successfully generated notebook at {output_path}")
