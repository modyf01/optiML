"""Edge AI showcase: Iris flower classifier — OptiML vs gradient descent.

Dataset
-------
Fisher's Iris (1936) — a REAL botanical dataset with 150 samples.
We classify all three species (setosa, versicolor, virginica) from
all four measurements (sepal length/width, petal length/width).

Architecture
------------
Two-layer ReLU network: Linear → ReLU → Linear.
Targets are one-hot encoded; predictions use argmax.
Inputs scaled with MinMaxScaler to [0, 1].

PyTorch:  fixed 3-neuron hidden layer, trained with Adam (local search).
OptiML:   convex reformulation (Pilanci & Ergen, ICML 2020).
          With exact=True: enumerate ALL sign patterns →
          single convex SOCP → **certified global optimum** (Theorem 1).
          No binary variables — the entire problem is convex.
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

SEED = 42
TRAIN_SIZE = 0.7

GD_RESTARTS = 10
GD_MAX_EPOCHS = 5000
GD_LR = 0.02
GD_PATIENCE = 50
GD_MIN_DELTA = 1e-6

BETA = 0.001
N_PATTERNS = 50


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
    X_raw = iris.data
    y_int = iris.target
    y_onehot = np.eye(3)[y_int]

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
    print("  Activation: ReLU (exact)")
    print(f"  Train: {n_train}  |  Test: {len(X_test)}")
    print("=" * 62)

    # ── Round 1: PyTorch + Adam + early stopping ──────────────────────
    print(f"\n[Round 1] PyTorch Adam — {GD_RESTARTS} random restarts"
          f" (Linear(4,3) → ReLU → Linear(3,3), "
          f"patience={GD_PATIENCE})\n")

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

    # ── Round 2: OptiML Convex SOCP (certified global optimum) ────────
    print(f"\n[Round 2] OptiML ConvexReLUNet — "
          f"exact (all patterns, pure convex SOCP, β={BETA})\n")

    convex_net = optiml.ConvexReLUNet(
        in_features=4, out_features=3, n_patterns=N_PATTERNS,
    )

    t0 = time.time()
    convex_net.fit(X_train, y_train_oh, beta=BETA, tee=False,
                   exact=True)
    solve_time = time.time() - t0

    optiml_pt = convex_net.export('pytorch')
    n_hidden = optiml_pt[0].out_features
    n_params = sum(p.numel() for p in optiml_pt.parameters())

    criterion = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        optiml_sse = criterion(optiml_pt(X_tr_t), y_tr_t).item()
    optiml_mse = optiml_sse / n_train
    optiml_acc = evaluate(optiml_pt, X_te_t, y_te_t)

    cert_tag = "CERTIFIED" if convex_net.certified else "upper bound"
    print(f"\n  Hidden neurons: {n_hidden}  |  Params: {n_params}")
    print(f"  MSE = {optiml_mse:.6f}  (SSE = {optiml_sse:.6f})")
    print(f"  Status: {cert_tag}")
    print(f"  Solved in {solve_time:.1f}s")

    # ── Round 3: OptiML Deep (3-layer, 2 hidden) ─────────────────────
    print(f"\n[Round 3] OptiML DeepConvexReLUNet — "
          f"3-layer SOCP (width=3, P=5, β={BETA})\n")

    deep_net = optiml.DeepConvexReLUNet(
        in_features=4, out_features=3, hidden_layers=2,
        width=3, n_patterns=5,
    )

    t0 = time.time()
    deep_net.fit(X_train, y_train_oh, beta=BETA, tee=False)
    deep_time = time.time() - t0

    raw_preds = deep_net.predict_convex(X_train)
    raw_train_acc = (raw_preds.argmax(1) == y_train_oh.argmax(1)
                     ).mean() * 100

    raw_test = deep_net.predict_convex(X_test)
    raw_test_acc = (raw_test.argmax(1) == y_test_int).mean() * 100

    deep_pt = deep_net.export('pytorch')
    deep_pt.eval()

    with torch.no_grad():
        pt_preds = deep_pt(X_tr_t).numpy()
        pt_test_preds = deep_pt(X_te_t).numpy()

    pt_train_acc = (pt_preds.argmax(1) == y_train_oh.argmax(1)
                    ).mean() * 100
    deep_acc = (pt_test_preds.argmax(1) == y_test_int).mean() * 100

    diff_train = np.abs(raw_preds - pt_preds).max()
    diff_test = np.abs(raw_test - pt_test_preds).max()

    print(f"  Obj = {deep_net.objective_value:.6f}")
    print(f"  Raw convex  train acc: {raw_train_acc:.1f}%  "
          f"test acc: {raw_test_acc:.1f}%")
    print(f"  PT export   train acc: {pt_train_acc:.1f}%  "
          f"test acc: {deep_acc:.1f}%")
    print(f"  max|diff| train={diff_train:.4f}  "
          f"test={diff_test:.4f}")
    print(f"  Solved in {deep_time:.1f}s")

    # ── Results ───────────────────────────────────────────────────────
    best_gd_mse = gd_losses[best_idx] / n_train
    median_gd_mse = np.median(gd_losses) / n_train
    first_gd_mse = gd_losses[0] / n_train

    with torch.no_grad():
        deep_sse = criterion(deep_pt(X_tr_t), y_tr_t).item()
    deep_mse = deep_sse / n_train

    print("\n" + "=" * 62)
    print(f"  RESULTS — test accuracy on {len(X_test)} unseen iris samples")
    print("=" * 62)
    print(f"  {'Method':<30} {'Train MSE':>10} {'Test Acc':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    cert2 = "GLOBAL OPT" if convex_net.certified else "UB"
    print(f"  {f'OptiML 2-layer ({cert2})':<30}"
          f" {optiml_mse:>10.4f} {optiml_acc:>9.1f}%")
    print(f"  {'OptiML 3-layer (UB)':<30}"
          f" {deep_mse:>10.4f} {deep_acc:>9.1f}%")
    print(f"  {'GD best of ' + str(GD_RESTARTS):<30}"
          f" {best_gd_mse:>10.4f} {gd_accs[best_idx]:>9.1f}%")
    print(f"  {'GD median':<30}"
          f" {median_gd_mse:>10.4f} {np.median(gd_accs):>9.1f}%")
    print(f"  {'GD single shot (seed=0)':<30}"
          f" {first_gd_mse:>10.4f} {gd_accs[0]:>9.1f}%")
    print("=" * 62)

    if convex_net.certified:
        print(f"\n  OptiML 2-layer: CERTIFIED GLOBAL OPTIMUM "
              f"(all patterns enumerated, pure convex SOCP)")
    else:
        print(f"\n  OptiML 2-layer: upper bound on the global "
              f"optimum of ReLU training (sampled patterns)")
    print(f"  2-layer: {n_hidden} hidden neurons, {n_params} params, "
          f"solved in {solve_time:.1f}s.")
    print(f"  3-layer: solved in {deep_time:.1f}s (upper bound).")
    print(f"  Ref: Pilanci & Ergen ICML 2020 / "
          f"Ergen & Pilanci ICML 2021\n")


if __name__ == "__main__":
    main()
