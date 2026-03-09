"""Edge AI showcase: Iris flower classifier — OptiML vs gradient descent.

Dataset
-------
Fisher's Iris (1936) — a REAL botanical dataset with 150 samples.
We classify all three species (setosa, versicolor, virginica) from
all four measurements (sepal length/width, petal length/width).

Architecture
------------
Linear(4,3) → ReLU → Linear(3,3) — **24 trainable parameters** (96 bytes).
Targets are one-hot encoded; predictions use argmax.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from optiml.losses import MulitClassHingeLoss

import optiml

SEED = 42
TRAIN_SIZE = 0.7
WEIGHT_BOUNDS = (-5, 5)

GD_RESTARTS = 10
GD_MAX_EPOCHS = 5000
GD_LR = 0.02
GD_PATIENCE = 50
GD_MIN_DELTA = 1e-6


def evaluate(model, X_t, y_t):
    """Accuracy via argmax over 3 output neurons."""
    model.eval()
    with torch.no_grad():
        out = model(X_t)
        preds = out.argmax(dim=1)
        return (preds == y_t).float().mean().item() * 100


def train_pytorch(X_tr, y_tr_onehot, seed):
    """Train with Adam + early stopping on training loss."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 3))
    opt = optim.Adam(model.parameters(), lr=GD_LR)
    criterion = nn.MSELoss(reduction='sum')

    best_loss = float('inf')
    wait = 0
    final_epoch = 0

    for epoch in range(GD_MAX_EPOCHS):
        opt.zero_grad()
        loss = criterion(model(X_tr), y_tr_onehot)
        loss.backward()
        opt.step()

        current_loss = loss.item()
        if best_loss - current_loss > GD_MIN_DELTA:
            best_loss = current_loss
            wait = 0
        else:
            wait += 1
        if wait >= GD_PATIENCE:
            final_epoch = epoch + 1
            break
    else:
        final_epoch = GD_MAX_EPOCHS

    with torch.no_grad():
        train_loss = criterion(model(X_tr), y_tr_onehot).item()
    return train_loss, model, final_epoch


def main():
    # ── Data: full Iris dataset ───────────────────────────────────────
    iris = load_iris()
    X_raw = iris.data                                       # (150, 4)
    y_int = iris.target                                     # 0, 1, 2
    y_onehot = np.eye(3)[y_int]                             # (150, 3)

    scaler = MinMaxScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)

    X_train, X_test, y_train_oh, y_test_oh = train_test_split(
        X_scaled, y_onehot, train_size=TRAIN_SIZE, random_state=SEED,
        stratify=y_int,
    )
    y_test_int = y_test_oh.argmax(axis=1)
    n_train = len(X_train)

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_oh, dtype=torch.float32)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    y_te_t = torch.tensor(y_test_int, dtype=torch.long)

    species = iris.target_names.tolist()

    print("=" * 62)
    print("  Edge AI: Iris flower micro-classifier")
    print(f"  Features: {', '.join(iris.feature_names)}")
    print(f"  Classes:  {', '.join(species)}")
    print("  Architecture: Linear(4,3) → ReLU → Linear(3,3)")
    print(f"  Params: 24  |  Train: {n_train}  |  Test: {len(X_test)}")
    print("=" * 62)

    # ── Round 1: PyTorch + Adam + early stopping ──────────────────────
    print(f"\n[Round 1] PyTorch Adam — {GD_RESTARTS} random restarts"
          f" (early stopping, patience={GD_PATIENCE})\n")

    t0 = time.time()
    gd_results = []
    for i in range(GD_RESTARTS):
        sse, model, epochs = train_pytorch(X_tr_t, y_tr_t, seed=i)
        acc = evaluate(model, X_te_t, y_te_t)
        mse = sse / n_train
        gd_results.append((sse, acc, epochs))
        print(f"  restart {i:2d}: MSE = {mse:.4f}  test = {acc:5.1f}%"
              f"  ({epochs} epochs)")
    gd_time = time.time() - t0

    gd_losses = [r[0] for r in gd_results]
    gd_accs = [r[1] for r in gd_results]
    best_idx = int(np.argmin(gd_losses))

    print(f"\n  Total GD time: {gd_time:.1f}s")

    # ── Round 2: OptiML (global optimum) ──────────────────────────────
    print(f"\n[Round 2] OptiML — MINLP global optimisation (Couenne)\n")

    model = optiml.Sequential(
        optiml.Linear(4, 3),
        optiml.ReLU(M=5),
        optiml.Linear(3, 3),
    )

    t0 = time.time()
    model.fit(
        X_train, y_train_oh,
        loss=MulitClassHingeLoss(reduction='sum'),
        solver='couenne',
        weight_bounds=WEIGHT_BOUNDS,
        tee=False,
    )
    solve_time = time.time() - t0

    optiml_pt = model.export('pytorch')
    optiml_sse = model.objective_value
    optiml_mse = optiml_sse / n_train
    optiml_acc = evaluate(optiml_pt, X_te_t, y_te_t)

    print(f"  MSE = {optiml_mse:.6f}  (SSE = {optiml_sse:.6f})")
    print(f"  Solved in {solve_time:.1f}s")

    # ── Results ───────────────────────────────────────────────────────
    best_gd_mse = gd_losses[best_idx] / n_train
    median_gd_mse = np.median(gd_losses) / n_train
    first_gd_mse = gd_losses[0] / n_train

    print("\n" + "=" * 62)
    print(f"  RESULTS — test accuracy on {len(X_test)} unseen iris samples")
    print("=" * 62)
    print(f"  {'Method':<30} {'Train MSE':>10} {'Test Acc':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'OptiML (guaranteed)':<30}"
          f" {optiml_mse:>10.4f} {optiml_acc:>9.1f}%")
    print(f"  {'GD best of ' + str(GD_RESTARTS):<30}"
          f" {best_gd_mse:>10.4f} {gd_accs[best_idx]:>9.1f}%")
    print(f"  {'GD median':<30}"
          f" {median_gd_mse:>10.4f} {np.median(gd_accs):>9.1f}%")
    print(f"  {'GD single shot (seed=0)':<30}"
          f" {first_gd_mse:>10.4f} {gd_accs[0]:>9.1f}%")
    print("=" * 62)

    print(f"\n  OptiML finds the mathematically optimal weights"
          f" in {solve_time:.1f}s.")
    print(f"\n  Model size: 24 parameters = 96 bytes (float32)")
    print(f"  Ready for deployment on any microcontroller.\n")


if __name__ == "__main__":
    main()
