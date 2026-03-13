"""Quick test: verify exact=True (all patterns, pure convex SOCP) on a tiny dataset."""
import os
os.environ['GRB_WLSACCESSID'] = '7999b455-6a85-40b2-aea2-1cd45f4c68ec'
os.environ['GRB_WLSSECRET'] = 'b9bbb913-2b9e-4a8c-bf60-4e1f0b1c5ded'
os.environ['GRB_LICENSEID'] = '2790219'

import numpy as np
from optiml import ConvexReLUNet

np.random.seed(0)
n, d = 10, 2
X = np.random.randn(n, d)
w_true = np.array([1.0, -0.5])
y = np.maximum(X @ w_true, 0) + 0.1 * np.random.randn(n)
y = y.reshape(-1, 1)

print(f"n={n}, d={d}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

model = ConvexReLUNet(in_features=d, out_features=1, n_patterns=10)
model.fit(X, y, beta=0.01, tee=False, verbose=True, exact=True)

print(f"\nCertified: {model.certified}")
print(f"Objective: {model.objective_value:.8f}")

pt_model = model.export('pytorch')
import torch
with torch.no_grad():
    X_t = torch.tensor(X, dtype=torch.float32)
    pred_pt = pt_model(X_t).numpy().flatten()

print(f"\nPyTorch predictions vs y:")
for i in range(n):
    print(f"  y={y[i,0]:.4f}  pred={pred_pt[i]:.4f}")
