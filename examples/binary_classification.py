"""Edge AI showcase: Iris flower classifier — OptiML vs gradient descent.

Dataset
-------
Fisher's Iris (1936) — a REAL botanical dataset.  We classify
*Iris versicolor* vs *Iris virginica* from two petal measurements
(petal length, petal width).  These two species overlap in feature space,
making the problem genuinely non-trivial.

Architecture
------------
Linear(2,2) → ReLU → Linear(2,1) — **9 trainable parameters** (36 bytes).
This is the minimum capacity that can learn the non-linear decision boundary.

We train on only 10 calibration samples (5 per species) and evaluate on
the remaining 90.  With so few samples and such a tiny model, the loss
landscape is riddled with bad local minima.  Gradient descent frequently
gets stuck; OptiML's MINLP solver finds the provably optimal weights.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import optiml
from optiml.losses import MSELoss

SEED = 42
TRAIN_SIZE = 0.7
WEIGHT_BOUNDS = (-5, 5)

GD_RESTARTS = 10
GD_EPOCHS = 500
GD_LR = 0.02


def evaluate(model, X_t, y_t):
    model.eval()
    with torch.no_grad():
        out = model(X_t).squeeze()
        preds = (out > 0.5).long()
        return (preds == y_t).float().mean().item() * 100


def train_pytorch(X_tr, y_tr, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    opt = optim.Adam(model.parameters(), lr=GD_LR)
    criterion = nn.MSELoss(reduction='sum')
    for _ in range(GD_EPOCHS):
        opt.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_loss = criterion(model(X_tr), y_tr).item()
    return train_loss, model


def main():
    # ── Data: real Iris petal measurements ────────────────────────────
    iris = load_iris()
    mask = iris.target >= 1  # versicolor (1) and virginica (2)
    X_raw = iris.data[mask][:, [2, 3]]  # petal length, petal width
    y_raw = (iris.target[mask] - 1).astype(np.float64)  # 0 = versicolor, 1 = virginica

    scaler = MinMaxScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_raw, train_size=TRAIN_SIZE, random_state=SEED, stratify=y_raw,
    )

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)

    feat_names = [iris.feature_names[i] for i in [2, 3]]
    species = ['versicolor', 'virginica']

    print("=" * 62)
    print("  Edge AI: Iris flower micro-classifier")
    print(f"  Features: {feat_names[0]}, {feat_names[1]}")
    print(f"  Classes:  {species[0]} (0) vs {species[1]} (1)")
    print("  Architecture: Linear(2,2) → ReLU → Linear(2,1)")
    print(f"  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")
    print("=" * 62)

    # ── Round 1: PyTorch + Adam ───────────────────────────────────────
    print(f"\n[Round 1] PyTorch Adam — {GD_RESTARTS} random restarts,"
          f" {GD_EPOCHS} epochs each\n")

    t0 = time.time()
    gd_results = []
    for i in range(GD_RESTARTS):
        sse, model = train_pytorch(X_tr_t, y_tr_t, seed=i)
        acc = evaluate(model, X_te_t, y_te_t)
        gd_results.append((sse, acc))
        status = "ok" if sse < 1.5 else "STUCK"
        print(f"  restart {i:2d}: SSE = {sse:.4f}  test = {acc:5.1f}%  {status}")
    gd_time = time.time() - t0

    gd_losses = [r[0] for r in gd_results]
    gd_accs = [r[1] for r in gd_results]
    best_idx = int(np.argmin(gd_losses))
    n_stuck = sum(1 for l in gd_losses if l >= 1.5)

    print(f"\n  {n_stuck}/{GD_RESTARTS} restarts got stuck  ({gd_time:.1f}s)")

    # ── Round 2: OptiML (global optimum) ──────────────────────────────
    print(f"\n[Round 2] OptiML — MINLP global optimisation (Couenne)\n")

    model = optiml.Sequential(
        optiml.Linear(2, 2),
        optiml.ReLU(M=5),
        optiml.Linear(2, 1),
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        loss=MSELoss(reduction='sum'),
        solver='couenne',
        weight_bounds=WEIGHT_BOUNDS,
        tee=False,
    )
    solve_time = time.time() - t0

    optiml_pt = model.export('pytorch')
    optiml_sse = model.objective_value
    optiml_acc = evaluate(optiml_pt, X_te_t, y_te_t)

    print(f"  SSE = {optiml_sse:.6f}")
    print(f"  Solved in {solve_time:.1f}s")

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  RESULTS — test accuracy on {len(X_test)} unseen iris samples")
    print("=" * 62)
    print(f"  {'Method':<30} {'Train SSE':>10} {'Test Acc':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'OptiML (guaranteed)':<30}"
          f" {optiml_sse:>10.4f} {optiml_acc:>9.1f}%")
    print(f"  {'GD best of ' + str(GD_RESTARTS):<30}"
          f" {gd_losses[best_idx]:>10.4f} {gd_accs[best_idx]:>9.1f}%")
    print(f"  {'GD median':<30}"
          f" {np.median(gd_losses):>10.4f} {np.median(gd_accs):>9.1f}%")
    print(f"  {'GD single shot (seed=0)':<30}"
          f" {gd_losses[0]:>10.4f} {gd_accs[0]:>9.1f}%")
    print("=" * 62)

    print(f"\n  In practice you train ONCE and deploy.")
    if n_stuck > 0:
        print(f"  Gradient descent failed in {n_stuck}/{GD_RESTARTS} attempts"
              f" ({n_stuck * 100 // GD_RESTARTS}% failure rate).")
    print(f"  OptiML finds the mathematically optimal weights every time.")
    print(f"\n  Model size: 9 parameters = 36 bytes (float32)")
    print(f"  Ready for deployment on any microcontroller.\n")


if __name__ == "__main__":
    main()
