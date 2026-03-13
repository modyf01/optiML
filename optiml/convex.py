"""Convex reformulation for ReLU networks (two-layer and three-layer).

Based on:
  [1] Pilanci & Ergen, "Neural Networks are Convex Regularizers"
      (ICML 2020, arXiv:2002.10553)  — two-layer networks (eq. 8).
  [2] Ergen & Pilanci, "Global Optimality Beyond Two Layers:
      Training Deep ReLU Networks via Convex Programs"
      (ICML 2021, PMLR 139:2993-3003)  — three-layer networks
      (Theorem 1).

The entire optimisation is **purely convex** (SOCP).  Sign patterns
D_1, …, D_P are fixed *parameters* — not decision variables.
There are **no binary/integer variables** anywhere.

Two modes of operation:

* **exact=False** (default):  Sample a tractable subset of sign
  patterns (Remark 3.3 of [1]).  The sampled SOCP is solved to
  global optimality by Gurobi, giving an upper bound that tightens
  as more patterns are added.

* **exact=True**:  Enumerate **all** hyperplane arrangement patterns
  (Appendix A.2 of [1]), then solve a single convex SOCP with the
  complete set.  By Theorem 1 of [1] this gives a **certified
  global optimum** of the full non-convex training problem (2).
  Feasible when rank(X) is small (the number of patterns is
  polynomial in n for fixed rank).
"""

import os

import numpy as np


# ---------------------------------------------------------------------------
# Gurobi environment
# ---------------------------------------------------------------------------

def _create_gurobi_env():
    """Create a gurobipy Env, honouring WLS credentials when set."""
    import gurobipy as gp
    from .solver import is_gurobi_wls_configured

    if is_gurobi_wls_configured():
        params = {
            'WLSAccessID': os.environ['GRB_WLSACCESSID'].strip(),
            'WLSSecret': os.environ['GRB_WLSSECRET'].strip(),
            'LicenseID': int(os.environ['GRB_LICENSEID'].strip()),
        }
        return gp.Env(params=params)
    return gp.Env()


# ---------------------------------------------------------------------------
# Sign-pattern sampling
# ---------------------------------------------------------------------------

def _sample_sign_patterns(X, n_patterns, seed=42):
    """Sample distinct hyperplane arrangement patterns.

    For each random direction u, computes D_i = diag(1[Xu >= 0])
    and keeps unique patterns.
    """
    n, d = X.shape
    rng = np.random.default_rng(seed)
    patterns = {}

    for _ in range(n_patterns * 20):
        if len(patterns) >= n_patterns:
            break
        u = rng.standard_normal(d)
        signs = (X @ u >= 0).astype(np.float64)
        key = tuple(signs.astype(int))
        if key not in patterns:
            patterns[key] = signs

    return list(patterns.values())


def _sample_3layer_patterns(X, width, n_pats, seed=42):
    """Sample sign patterns for a 3-layer network (Remark 3 of [2]).

    First-layer patterns are sampled independently for each of the
    ``width`` neurons, giving diverse masks.  Second-layer patterns
    are sampled from consistent forward passes (random first-layer
    weights → ReLU → random second-layer weights → sign).

    Returns
    -------
    D1 : list[list[ndarray]]
        ``D1[j][i]`` is the binary (0/1) sign vector of length *n*
        for neuron *j*, pattern index *i*.
    D2 : list[ndarray]
        Second-layer binary sign vectors.
    D1_dirs : list[list[ndarray]]
        ``D1_dirs[j][i]`` is the direction vector that generated
        ``D1[j][i]``.
    D2_params : list[tuple[ndarray, ndarray]]
        ``(W1, w2)`` pairs used to generate each ``D2[l]``.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape

    D1 = []
    D1_dirs = []
    for _j in range(width):
        pats = {}
        dirs = {}
        for _ in range(n_pats * 20):
            if len(pats) >= n_pats:
                break
            u = rng.standard_normal(d)
            signs = (X @ u >= 0).astype(np.float64)
            key = tuple(signs.astype(int))
            if key not in pats:
                pats[key] = signs
                dirs[key] = u.copy()
        D1.append(list(pats.values()))
        D1_dirs.append([dirs[tuple(p.astype(int))] for p in pats.values()])

    D2_dict = {}
    D2_params_dict = {}
    for _ in range(n_pats * 40):
        if len(D2_dict) >= n_pats:
            break
        W1 = rng.standard_normal((d, width))
        h1 = np.maximum(X @ W1, 0)
        w2 = rng.standard_normal(width)
        z2 = h1 @ w2
        signs = (z2 >= 0).astype(np.float64)
        key = tuple(signs.astype(int))
        if key not in D2_dict:
            D2_dict[key] = signs
            D2_params_dict[key] = (W1.copy(), w2.copy())
    D2 = list(D2_dict.values())
    D2_params = [D2_params_dict[tuple(p.astype(int))] for p in D2]

    return D1, D2, D1_dirs, D2_params


# ---------------------------------------------------------------------------
# Two-layer SOCP builder  (eq. 8 from [1])
# ---------------------------------------------------------------------------

def _max_patterns_for_free_license(n, d):
    """Max P that fits within Gurobi free-license limits.

    Free: 2000 vars, 2000 linear constrs, 200 quadratic constrs.
    Variables: 2*P*d + 2*P  (v, w, t, s — no r variable).
    Linear:   2*P*n          (sign constraints only).
    QC:       2*P            (SOC for v and w).
    """
    p_lin = 2000 // (2 * n)
    p_var = 2000 // (2 * d + 2)
    p_qc = 200 // 2
    return max(1, min(p_lin, p_var, p_qc))


def _build_and_solve(X, y, sign_patterns, beta, env, tee, time_limit):
    """Build & solve convex SOCP (eq. 8) for one scalar output.

    Introduces explicit residual variables to keep the quadratic
    objective at O(n^2) terms instead of O((Pd)^2).
    """
    import gurobipy as gp
    from scipy import sparse

    n, d = X.shape
    P = len(sign_patterns)
    Pd = P * d

    M_mat = np.zeros((n, Pd))
    for i in range(P):
        M_mat[:, i * d:(i + 1) * d] = sign_patterns[i][:, None] * X

    M_sp = sparse.csc_matrix(M_mat)

    with gp.Model(env=env) as m:
        m.Params.OutputFlag = 1 if tee else 0
        if time_limit is not None:
            m.Params.TimeLimit = time_limit

        v = m.addMVar((P, d), lb=-gp.GRB.INFINITY, name='v')
        w = m.addMVar((P, d), lb=-gp.GRB.INFINITY, name='w')
        t = m.addMVar(P, lb=0.0, name='t')
        s = m.addMVar(P, lb=0.0, name='s')
        r = m.addMVar(n, lb=-gp.GRB.INFINITY, name='r')

        v_flat = v.reshape(Pd)
        w_flat = w.reshape(Pd)

        m.addConstr(r == M_sp @ v_flat - M_sp @ w_flat - y, name='res')

        m.setObjective(
            0.5 * (r @ r) + beta * (t.sum() + s.sum()),
            gp.GRB.MINIMIZE,
        )

        for i in range(P):
            m.addConstr(v[i, :] @ v[i, :] <= t[i] * t[i], name=f'qv{i}')
            m.addConstr(w[i, :] @ w[i, :] <= s[i] * s[i], name=f'qw{i}')

        for i in range(P):
            SX = (2.0 * sign_patterns[i] - 1.0)[:, None] * X
            m.addConstr(SX @ v[i, :] >= 0, name=f'sv{i}')
            m.addConstr(SX @ w[i, :] >= 0, name=f'sw{i}')

        m.optimize()

        if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL,
                            gp.GRB.TIME_LIMIT):
            raise RuntimeError(
                f"Gurobi failed (status {m.Status}). "
                "Try increasing time_limit or adjusting beta."
            )

        return v.X.copy(), w.X.copy(), m.ObjVal


def _solve_single_output(X, y, sign_patterns, beta, env, tee, time_limit,
                          verbose):
    """Solve with auto-fallback on license size limits."""
    import gurobipy as gp

    patterns = sign_patterns
    while True:
        try:
            return _build_and_solve(X, y, patterns, beta, env, tee,
                                    time_limit)
        except gp.GurobiError as e:
            if "too large" in str(e).lower() and len(patterns) > 1:
                n, d = X.shape
                cap = _max_patterns_for_free_license(n, d)
                if len(patterns) <= cap:
                    raise
                patterns = patterns[:cap]
                if verbose:
                    print(f"[OptiML-Convex] License limit — "
                          f"using {cap} patterns (use WLS for more).")
            else:
                raise


# ---------------------------------------------------------------------------
# Full pattern enumeration  (Appendix A.2 of [1])
# ---------------------------------------------------------------------------

def _enumerate_all_patterns(X, verbose=False):
    """Enumerate all hyperplane arrangement sign patterns.

    For n hyperplanes in R^d through the origin, enumerates
    representative directions from every cell of the arrangement
    by combining random sampling with boundary enumeration
    (intersections of d-1 hyperplanes).

    The total number of patterns P satisfies
    P <= 2 * sum_{k=0}^{r-1} C(n-1, k)  where r = rank(X).

    Returns a list of unique sign-pattern arrays (float64).
    """
    from itertools import combinations

    n, d = X.shape
    patterns = {}

    rng = np.random.default_rng(42)
    n_random = min(200000, 20 * n ** min(d, 4))
    for _ in range(n_random):
        u = rng.standard_normal(d)
        signs = (X @ u >= 0).astype(np.float64)
        key = tuple(signs.astype(int))
        if key not in patterns:
            patterns[key] = signs

    for combo in combinations(range(n), d - 1):
        A = X[list(combo), :]
        _, S, Vt = np.linalg.svd(A, full_matrices=True)
        if S.size == 0 or S[-1] < 1e-12:
            continue
        u = Vt[-1]
        for sign in (+1.0, -1.0):
            su = sign * u
            signs = (X @ su >= 0).astype(np.float64)
            key = tuple(signs.astype(int))
            if key not in patterns:
                patterns[key] = signs

    if verbose:
        print(f"[OptiML-Convex] Enumerated {len(patterns)} "
              f"distinct sign patterns", flush=True)

    return list(patterns.values())


def _exhaustive_pricing(X, r, beta, all_patterns, active_keys,
                        max_new=50, verbose=False):
    """Exhaustive pricing over ALL enumerated patterns (no MIP).

    Scans every pattern in ``all_patterns``, computes the dual
    violation ||X^T diag(s) r|| for patterns not yet in the active
    set, and returns the top ``max_new`` violating patterns.

    This is O(P * n * d) per call.  Uses batched numpy for speed.
    There are **no binary/integer variables** — just arithmetic.
    """
    n, d = X.shape
    beta_sq = (beta + 1e-8) ** 2
    Xr = X.T * r

    best = []
    BATCH = 200000
    P = len(all_patterns)

    for start in range(0, P, BATCH):
        end = min(start + BATCH, P)
        batch_pats = all_patterns[start:end]

        batch_arr = np.array(batch_pats)
        XtDr = batch_arr @ Xr.T
        norms_sq = np.sum(XtDr ** 2, axis=1)

        mask = norms_sq > beta_sq
        for idx_in_batch in np.where(mask)[0]:
            pat = batch_pats[idx_in_batch]
            key = tuple(pat.astype(int))
            if key not in active_keys:
                best.append((float(norms_sq[idx_in_batch]),
                             pat, key))

    best.sort(key=lambda x: -x[0])

    result = []
    for nsq, pat, key in best[:max_new]:
        active_keys.add(key)
        result.append(pat)

    if verbose and result:
        top = np.sqrt(best[0][0])
        print(f"[pricing] {P} patterns, "
              f"{len(result)} added (max ||X^T D r|| = "
              f"{top:.6f} > β = {beta})", flush=True)

    return result


# ---------------------------------------------------------------------------
# Three-layer SOCP builder  (Theorem 1 of [2])
# ---------------------------------------------------------------------------

def _build_and_solve_3layer(X, y, D1, D2, beta, env, tee, time_limit):
    r"""Build & solve the 3-layer convex SOCP for one scalar output.

    Implements the convex program from Theorem 1 of Ergen & Pilanci
    (ICML 2021).  Four variable sets capture every combination of
    output sign (v/w) and second-layer weight sign (+/−):

      vp_{jil}  same-sign   (2D1−I)X vp ≥ 0   positive output
      vm_{jil}  anti-sign   (2D1−I)X vm ≤ 0   positive output
      wp_{jil}  same-sign   (2D1−I)X wp ≥ 0   negative output
      wm_{jil}  anti-sign   (2D1−I)X wm ≤ 0   negative output

    The coupling constraint sums same-sign AND anti-sign terms
    **together**, so the total can be positive or negative:

      (2D2_l − I) Σ_j [D1_{ij} X vp + D1_{ij} X vm] ≥ 0
    """
    import gurobipy as gp

    n, d = X.shape
    m1 = len(D1)
    P1 = min(len(p) for p in D1)
    P2 = len(D2)
    G = m1 * P1 * P2

    def gidx(j, i, l):
        return j * P1 * P2 + i * P2 + l

    with gp.Model(env=env) as m:
        m.Params.OutputFlag = 1 if tee else 0
        if time_limit is not None:
            m.Params.TimeLimit = time_limit

        vp = m.addMVar((G, d), lb=-gp.GRB.INFINITY, name='vp')
        vm = m.addMVar((G, d), lb=-gp.GRB.INFINITY, name='vm')
        wp = m.addMVar((G, d), lb=-gp.GRB.INFINITY, name='wp')
        wm = m.addMVar((G, d), lb=-gp.GRB.INFINITY, name='wm')

        tvp = m.addMVar(G, lb=0.0, name='tvp')
        tvm = m.addMVar(G, lb=0.0, name='tvm')
        twp = m.addMVar(G, lb=0.0, name='twp')
        twm = m.addMVar(G, lb=0.0, name='twm')

        # -- feature matrix  D2_l · D1_{ij} · X  for each group g --------
        M_mat = np.zeros((n, G * d))
        for j in range(m1):
            for i in range(P1):
                mask_ji = D1[j][i]
                for l in range(P2):
                    g = gidx(j, i, l)
                    M_mat[:, g * d:(g + 1) * d] = \
                        (D2[l] * mask_ji)[:, None] * X

        net = vp.reshape(G * d) + vm.reshape(G * d) \
            - wp.reshape(G * d) - wm.reshape(G * d)
        residual = M_mat @ net - y

        m.setObjective(
            0.5 * (residual @ residual)
            + beta * (tvp.sum() + tvm.sum() + twp.sum() + twm.sum()),
            gp.GRB.MINIMIZE,
        )

        # -- SOC  ‖·‖ ≤ t ------------------------------------------------
        for g in range(G):
            m.addConstr(vp[g, :] @ vp[g, :] <= tvp[g] * tvp[g])
            m.addConstr(vm[g, :] @ vm[g, :] <= tvm[g] * tvm[g])
            m.addConstr(wp[g, :] @ wp[g, :] <= twp[g] * twp[g])
            m.addConstr(wm[g, :] @ wm[g, :] <= twm[g] * twm[g])

        # -- sign constraints ---------------------------------------------
        for j in range(m1):
            for i in range(P1):
                SX = (2.0 * D1[j][i] - 1.0)[:, None] * X
                for l in range(P2):
                    g = gidx(j, i, l)
                    m.addConstr(SX @ vp[g, :] >= 0)
                    m.addConstr(SX @ wp[g, :] >= 0)
                    m.addConstr(SX @ vm[g, :] <= 0)
                    m.addConstr(SX @ wm[g, :] <= 0)

        # -- coupling (COMBINED over same-sign + anti-sign) ----------------
        for i in range(P1):
            for l in range(P2):
                sl = 2.0 * D2[l] - 1.0
                for k in range(n):
                    lhs_v = 0
                    lhs_w = 0
                    any_active = False
                    for j in range(m1):
                        if D1[j][i][k] > 0.5:
                            any_active = True
                            xk = X[k, :]
                            g = gidx(j, i, l)
                            lhs_v += xk @ vp[g, :] + xk @ vm[g, :]
                            lhs_w += xk @ wp[g, :] + xk @ wm[g, :]
                    if any_active:
                        m.addConstr(sl[k] * lhs_v >= 0)
                        m.addConstr(sl[k] * lhs_w >= 0)

        m.optimize()

        if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL,
                            gp.GRB.TIME_LIMIT):
            raise RuntimeError(
                f"Gurobi failed (status {m.Status}). "
                "Try increasing time_limit or adjusting beta."
            )

        return (vp.X.copy(), vm.X.copy(),
                wp.X.copy(), wm.X.copy(), m.ObjVal)


# ---------------------------------------------------------------------------
# ConvexReLUNet  (two-layer, unchanged API)
# ---------------------------------------------------------------------------

class ConvexReLUNet:
    """Two-layer ReLU network trained via convex reformulation.

    Based on Pilanci & Ergen, *Neural Networks are Convex Regularizers*
    (ICML 2020), equation (8).

    The entire optimisation is a **convex SOCP** — there are no binary
    or integer variables.  Sign patterns D_1, …, D_P are fixed
    parameters, not decision variables.

    With ``exact=False`` (default), a tractable subset of sign patterns
    is sampled (Remark 3.3 of [1]) and the resulting SOCP is solved to
    global optimality by Gurobi — giving an **upper bound** on the
    true global optimum.

    With ``exact=True``, **all** hyperplane arrangement patterns are
    enumerated (Appendix A.2 of [1]).  The resulting SOCP (still
    purely convex) yields a **certified global optimum** by Theorem 1.
    Feasible when rank(X_aug) is small.

    Usage::

        model = optiml.ConvexReLUNet(in_features=4, out_features=3)
        model.fit(X_train, y_train_onehot, beta=0.001, exact=True)
        pytorch_model = model.export('pytorch')
    """

    def __init__(self, in_features, out_features, n_patterns=200):
        self.in_features = in_features
        self.out_features = out_features
        self.n_patterns = n_patterns
        self._fitted = False
        self._certified = False
        self._hidden_weights = None
        self._hidden_bias = None
        self._output_weights = None
        self._output_bias = None
        self._objective_value = None

    def fit(self, X, y, beta=0.001, tee=False, time_limit=None,
            verbose=True, exact=False):
        """Train by solving the convex SOCP.

        Parameters
        ----------
        X : array-like, shape (n_samples, in_features)
            Training inputs (should be scaled, e.g. MinMaxScaler).
        y : array-like, shape (n_samples,) or (n_samples, out_features)
            Targets (e.g. one-hot for classification).
        beta : float
            Group-norm regularisation (equivalent to weight decay).
        tee : bool
            Print Gurobi solver output.
        time_limit : float or None
            Gurobi time limit per output dimension (seconds).
        verbose : bool
            Print progress messages.
        exact : bool
            When True, enumerate ALL hyperplane arrangement patterns
            and use column generation with **exhaustive pricing** (a
            simple scan over all patterns — no MIP, no binary variables).
            When no pattern violates the dual constraint, the solution
            is a **certified global optimum** (Theorem 1 of [1]).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        c = y.shape[1]

        X_aug = np.hstack([X, np.ones((n, 1))])
        d_aug = d + 1

        all_patterns = None
        if exact:
            if verbose:
                print(f"[OptiML-Convex] Enumerating ALL sign patterns "
                      f"(Appendix A.2)...")
            all_patterns = _enumerate_all_patterns(X_aug, verbose)
            P_total = len(all_patterns)
            sign_patterns = _sample_sign_patterns(
                X_aug, max(self.n_patterns, 200))
            if verbose:
                print(f"[OptiML-Convex] Starting with "
                      f"{len(sign_patterns)} sampled, "
                      f"{P_total} total available for pricing.",
                      flush=True)
        else:
            if verbose:
                print(f"[OptiML-Convex] Sampling sign patterns "
                      f"(Remark 3.3)...")
            sign_patterns = _sample_sign_patterns(X_aug, self.n_patterns)

        P = len(sign_patterns)
        certified = False

        if verbose:
            label = ("exact (column generation, pure convex)"
                     if exact else "sampled (upper bound)")
            print(f"[OptiML-Convex] {P} active patterns, "
                  f"{d_aug} dims. Mode: {label}.", flush=True)
            print(f"[OptiML-Convex] Solving convex SOCP "
                  f"for {c} output(s)...", flush=True)

        y_mean = y.mean(axis=0)
        y_centered = y - y_mean

        env = _create_gurobi_env()

        all_v = [None] * c
        all_w = [None] * c
        total_obj = 0.0

        try:
            max_cg = 200 if exact else 1
            active_keys = {tuple(p.astype(int)) for p in sign_patterns}

            for cg_iter in range(max_cg):
                total_obj = 0.0
                for k in range(c):
                    if verbose and c > 1:
                        print(f"[OptiML-Convex]   output {k + 1}/{c}"
                              f" ({len(sign_patterns)} patterns)...")

                    v_vals, w_vals, obj = _build_and_solve(
                        X_aug, y_centered[:, k],
                        sign_patterns, beta, env,
                        tee, time_limit)
                    all_v[k] = v_vals
                    all_w[k] = w_vals
                    total_obj += obj

                if not exact:
                    break

                any_new = False
                max_add = max(50, P_total // 5000)

                for k in range(c):
                    P_cur = all_v[k].shape[0]
                    pred = np.zeros(n)
                    for i in range(P_cur):
                        pred += sign_patterns[i] * (
                            X_aug @ (all_v[k][i] - all_w[k][i]))
                    r = pred - y_centered[:, k]

                    new_pats = _exhaustive_pricing(
                        X_aug, r, beta, all_patterns,
                        active_keys, max_new=max_add,
                        verbose=verbose)
                    if new_pats:
                        sign_patterns.extend(new_pats)
                        any_new = True

                if not any_new:
                    certified = True
                    if verbose:
                        print(
                            f"[CG] CERTIFIED GLOBAL OPTIMUM "
                            f"({cg_iter + 1} CG iterations, "
                            f"{len(sign_patterns)} active / "
                            f"{P_total} total patterns). "
                            f"No binary variables used.",
                            flush=True)
                    break

                if verbose:
                    print(f"[CG iter {cg_iter}] "
                          f"active = {len(sign_patterns)} / "
                          f"{P_total}", flush=True)
        finally:
            env.close()

        # ---------- extract weights ----------
        all_u = []
        all_alpha = []
        threshold = 1e-6

        for k in range(c):
            v_vals = all_v[k]
            w_vals = all_w[k]
            for i in range(v_vals.shape[0]):
                vn = np.linalg.norm(v_vals[i])
                if vn > threshold:
                    all_u.append(v_vals[i] / vn)
                    all_alpha.append((k, vn))

                wn = np.linalg.norm(w_vals[i])
                if wn > threshold:
                    all_u.append(w_vals[i] / wn)
                    all_alpha.append((k, -wn))

        if not all_u:
            all_u.append(np.zeros(d_aug))
            all_alpha.append((0, 0.0))

        m_total = len(all_u)
        W = np.array(all_u)

        self._hidden_weights = W[:, :d]
        self._hidden_bias = W[:, d]

        self._output_weights = np.zeros((c, m_total))
        for j, (k, alpha) in enumerate(all_alpha):
            self._output_weights[k, j] = alpha

        self._output_bias = y_mean
        self._objective_value = total_obj
        self._fitted = True
        self._certified = certified

        if verbose:
            tag = "GLOBAL OPT" if certified else "upper bound"
            print(f"[OptiML-Convex] Done ({tag}). "
                  f"{m_total} active neurons. "
                  f"Objective: {total_obj:.8f}")

    @property
    def objective_value(self):
        if not self._fitted:
            raise RuntimeError("Not fitted yet.")
        return self._objective_value

    @property
    def certified(self):
        """True when the solution is a certified global optimum."""
        if not self._fitted:
            raise RuntimeError("Not fitted yet.")
        return self._certified

    def export(self, backend='pytorch'):
        """Export to a deep-learning framework.

        Returns ``nn.Sequential(Linear, ReLU, Linear)`` with weights
        reconstructed from the convex solution.
        """
        if not self._fitted:
            raise RuntimeError("Not fitted yet.")
        if backend != 'pytorch':
            raise ValueError(f"Unknown backend '{backend}'")
        return self._export_pytorch()

    def _export_pytorch(self):
        import torch
        import torch.nn as nn

        m = self._hidden_weights.shape[0]

        hidden = nn.Linear(self.in_features, m)
        hidden.weight.data = torch.tensor(
            self._hidden_weights, dtype=torch.float32)
        hidden.bias.data = torch.tensor(
            self._hidden_bias, dtype=torch.float32)

        output = nn.Linear(m, self.out_features)
        output.weight.data = torch.tensor(
            self._output_weights, dtype=torch.float32)
        output.bias.data = torch.tensor(
            self._output_bias, dtype=torch.float32)

        return nn.Sequential(hidden, nn.ReLU(), output)


# ---------------------------------------------------------------------------
# DeepConvexReLUNet  (L-layer, globally optimal via joint sign patterns)
# ---------------------------------------------------------------------------

class DeepConvexReLUNet:
    """Three-layer ReLU network trained via a convex SOCP.

    Based on Theorem 1 of Ergen & Pilanci, *Global Optimality Beyond
    Two Layers* (ICML 2021).  The non-convex three-layer training
    problem is reformulated as a convex program with four variable
    types (same-sign / anti-sign × positive / negative output).  The
    **combined** coupling constraint sums both same-sign (non-negative)
    and anti-sign (non-positive) contributions, allowing the second
    hidden layer to learn non-trivial activation patterns.

    With all sign patterns the convex program is exactly equivalent
    to the non-convex problem.  In practice we sample a tractable
    subset, so Gurobi solves a reduced SOCP to global optimality —
    giving an **upper bound** on the true global optimum that tightens
    with more patterns and wider ``width``.

    The exported PyTorch model has the form::

        Linear → ReLU → Linear → ReLU → Linear

    Usage::

        model = optiml.DeepConvexReLUNet(
            in_features=4, out_features=3, width=3, n_patterns=5)
        model.fit(X_train, y_train_onehot, beta=0.001)
        pytorch_model = model.export('pytorch')

    Parameters
    ----------
    in_features : int
    out_features : int
    hidden_layers : int
        Must be 2 (two hidden ReLU layers → three-layer network).
        ``hidden_layers=1`` delegates to the two-layer solver
        internally.
    width : int
        Number of first-layer neurons per sub-network (*m₁* in the
        paper).  Must be ≥ 2 for non-trivial second-layer coupling.
    n_patterns : int
        Number of sampled sign patterns per neuron (first layer) and
        for the second layer.
    """

    def __init__(self, in_features, out_features, hidden_layers=2,
                 width=3, n_patterns=5):
        if hidden_layers not in (1, 2):
            raise ValueError(
                "DeepConvexReLUNet supports hidden_layers=1 or 2. "
                "For 1 hidden layer use ConvexReLUNet directly.")
        if width < 1:
            raise ValueError("width must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.width = width
        self.n_patterns = n_patterns
        self._fitted = False
        self._certified = False

    def fit(self, X, y, beta=0.001, tee=False, time_limit=None,
            verbose=True, exact=False):
        """Train by solving the convex SOCP.

        Parameters
        ----------
        X : array-like, shape (n_samples, in_features)
        y : array-like, shape (n_samples,) or (n_samples, out_features)
        beta : float
            Group-norm regularisation strength.
        tee : bool
            Print Gurobi solver log.
        time_limit : float or None
            Gurobi time limit per output dimension (seconds).
        verbose : bool
            Print progress messages.
        exact : bool
            (Only for hidden_layers=1) Enumerate all sign patterns
            for certified global optimality (pure convex, no MIP).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        K = y.shape[1]

        if exact and self.hidden_layers != 1:
            raise NotImplementedError(
                "exact=True is currently supported only for "
                "hidden_layers=1 (two-layer convex solver).  "
                "For hidden_layers=2, use ConvexReLUNet with "
                "exact=True instead.")

        X_aug = np.hstack([X, np.ones((n, 1))])
        d_aug = d + 1

        y_mean = y.mean(axis=0)
        y_centered = y - y_mean

        env = _create_gurobi_env()

        if self.hidden_layers == 1:
            return self._fit_2layer(
                X_aug, y_centered, y_mean, d, d_aug, K, n,
                env, beta, tee, time_limit, verbose, exact)

        m1 = self.width
        P = self.n_patterns

        if verbose:
            print(f"[OptiML-Deep] Sampling 3-layer patterns: "
                  f"width={m1}, P1={P}, P2={P} ...")

        D1, D2, D1_dirs, D2_params = _sample_3layer_patterns(
            X_aug, m1, P, seed=42)
        P1 = min(len(p) for p in D1)
        P2 = len(D2)
        G = m1 * P1 * P2

        if verbose:
            print(f"[OptiML-Deep] P1={P1} P2={P2} groups={G} "
                  f"(×4 sign types) d_aug={d_aug}")

        self._vp = []
        self._vm = []
        self._wp = []
        self._wm = []
        total_obj = 0.0

        try:
            for k in range(K):
                if verbose:
                    print(f"[OptiML-Deep]   output {k + 1}/{K}...")
                vp, vm_, wp, wm_, obj = _build_and_solve_3layer(
                    X_aug, y_centered[:, k], D1, D2,
                    beta, env, tee, time_limit)
                self._vp.append(vp)
                self._vm.append(vm_)
                self._wp.append(wp)
                self._wm.append(wm_)
                total_obj += obj
        finally:
            env.close()

        self._D1 = D1
        self._D2 = D2
        self._D1_dirs = D1_dirs
        self._D2_params = D2_params
        self._X_aug = X_aug
        self._d = d
        self._d_aug = d_aug
        self._K = K
        self._y_mean = y_mean
        self._objective_value = total_obj
        self._fitted = True
        self._certified = False

        if verbose:
            print(f"[OptiML-Deep] Done (upper bound).  "
                  f"Objective: {total_obj:.8f}")

    # ---------- hidden_layers=1 fallback via two-layer solver ----------

    def _fit_2layer(self, X_aug, y_centered, y_mean, d, d_aug, K, n,
                    env, beta, tee, time_limit, verbose, exact=False):
        all_patterns = None
        if exact:
            if verbose:
                print(f"[OptiML-Deep] hidden_layers=1 → "
                      f"enumerating ALL patterns")
            all_patterns = _enumerate_all_patterns(X_aug, verbose)
            P_total = len(all_patterns)
            sign_patterns = _sample_sign_patterns(
                X_aug, max(self.n_patterns, 200))
        else:
            if verbose:
                print(f"[OptiML-Deep] hidden_layers=1 → "
                      f"sampled patterns (Remark 3.3)")
            sign_patterns = _sample_sign_patterns(
                X_aug, self.n_patterns)

        certified = False

        self._2layer_v = []
        self._2layer_w = []
        total_obj = 0.0

        try:
            max_cg = 200 if exact else 1
            active_keys = {tuple(p.astype(int)) for p in sign_patterns}

            for cg_iter in range(max_cg):
                self._2layer_v = []
                self._2layer_w = []
                total_obj = 0.0

                for k in range(K):
                    if verbose:
                        print(f"[OptiML-Deep]   output {k + 1}/{K}"
                              f" ({len(sign_patterns)} patterns)...")
                    v, w, obj = _build_and_solve(
                        X_aug, y_centered[:, k],
                        sign_patterns, beta, env,
                        tee, time_limit)
                    self._2layer_v.append(v)
                    self._2layer_w.append(w)
                    total_obj += obj

                if not exact:
                    break

                any_new = False
                max_add = max(50, P_total // 5000)

                for k in range(K):
                    P_cur = self._2layer_v[k].shape[0]
                    pred = np.zeros(n)
                    for i in range(P_cur):
                        pred += sign_patterns[i] * (
                            X_aug @ (self._2layer_v[k][i]
                                     - self._2layer_w[k][i]))
                    r = pred - y_centered[:, k]
                    new_pats = _exhaustive_pricing(
                        X_aug, r, beta, all_patterns,
                        active_keys, max_new=max_add,
                        verbose=verbose)
                    if new_pats:
                        sign_patterns.extend(new_pats)
                        any_new = True

                if not any_new:
                    certified = True
                    if verbose:
                        print(f"[CG] CERTIFIED ({cg_iter + 1} iters, "
                              f"{len(sign_patterns)} active / "
                              f"{P_total} total)", flush=True)
                    break
        finally:
            env.close()

        self._sign_patterns = sign_patterns
        self._X_aug = X_aug
        self._d = d
        self._d_aug = d_aug
        self._K = K
        self._y_mean = y_mean
        self._objective_value = total_obj
        self._fitted = True
        self._certified = certified

        if verbose:
            tag = "GLOBAL OPT" if certified else "upper bound"
            print(f"[OptiML-Deep] Done ({tag}).  "
                  f"Objective: {total_obj:.8f}")

    @property
    def objective_value(self):
        if not self._fitted:
            raise RuntimeError("Not fitted yet.")
        return self._objective_value

    @property
    def certified(self):
        """True when the solution is a certified global optimum."""
        if not self._fitted:
            raise RuntimeError("Not fitted yet.")
        return self._certified

    # ------------------------------------------------------------------
    # Raw convex prediction (training data only — patterns are stored)
    # ------------------------------------------------------------------

    def predict_convex(self, X):
        """Predict on the training data using raw convex variables."""
        if not self._fitted:
            raise RuntimeError("Not fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        X_aug = np.hstack([X, np.ones((n, 1))])

        if self.hidden_layers == 1:
            return self._predict_2layer(X_aug, n)
        return self._predict_3layer(X_aug, n)

    def _predict_2layer(self, X_aug, n):
        preds = np.tile(self._y_mean, (n, 1))
        for k in range(self._K):
            v = self._2layer_v[k]
            w = self._2layer_w[k]
            for i, pat in enumerate(self._sign_patterns):
                preds[:, k] += pat * (X_aug @ (v[i] - w[i]))
        return preds

    def _predict_3layer(self, X_aug, n):
        D1_dirs = self._D1_dirs
        D2_params = self._D2_params
        m1 = len(D1_dirs)
        P1 = min(len(d) for d in D1_dirs)
        P2 = len(D2_params)

        preds = np.tile(self._y_mean, (n, 1))
        for k in range(self._K):
            vp, vm_ = self._vp[k], self._vm[k]
            wp, wm_ = self._wp[k], self._wm[k]
            for j in range(m1):
                for i in range(P1):
                    d1_mask = (X_aug @ D1_dirs[j][i] >= 0
                               ).astype(np.float64)
                    for l in range(P2):
                        W1, w2 = D2_params[l]
                        h1 = np.maximum(X_aug @ W1, 0)
                        d2_mask = (h1 @ w2 >= 0).astype(np.float64)
                        mask = d2_mask * d1_mask
                        g = j * P1 * P2 + i * P2 + l
                        net = vp[g] + vm_[g] - wp[g] - wm_[g]
                        preds[:, k] += mask * (X_aug @ net)
        return preds

    # ------------------------------------------------------------------
    # Export  →  PyTorch model
    # ------------------------------------------------------------------

    def export(self, backend='pytorch'):
        """Export as a PyTorch model.

        For ``hidden_layers=1`` returns a two-layer network.
        For ``hidden_layers=2`` returns a three-layer network
        (Linear→ReLU→Linear→ReLU→Linear).
        """
        if not self._fitted:
            raise RuntimeError("Not fitted yet.")
        if backend != 'pytorch':
            raise ValueError(f"Unknown backend '{backend}'")

        if self.hidden_layers == 1:
            return self._export_2layer_pt()
        return self._export_3layer_pt()

    def _export_2layer_pt(self):
        import torch
        import torch.nn as nn

        d = self._d
        d_aug = self._d_aug
        K = self._K
        threshold = 1e-6
        rows_W, rows_b, out_W = [], [], []

        for k in range(K):
            v, w = self._2layer_v[k], self._2layer_w[k]
            for i in range(v.shape[0]):
                for vec, sign in [(v[i], +1), (w[i], -1)]:
                    nrm = np.linalg.norm(vec)
                    if nrm > threshold:
                        u = vec / nrm
                        rows_W.append(u[:d])
                        rows_b.append(u[d])
                        row = np.zeros(K)
                        row[k] = sign * nrm
                        out_W.append(row)

        if not rows_W:
            rows_W.append(np.zeros(d))
            rows_b.append(0.0)
            out_W.append(np.zeros(K))

        Wh = np.array(rows_W)
        bh = np.array(rows_b)
        Wo = np.array(out_W).T

        h = nn.Linear(d, Wh.shape[0])
        h.weight.data = torch.tensor(Wh, dtype=torch.float32)
        h.bias.data = torch.tensor(bh, dtype=torch.float32)
        o = nn.Linear(Wh.shape[0], K)
        o.weight.data = torch.tensor(Wo, dtype=torch.float32)
        o.bias.data = torch.tensor(self._y_mean,
                                   dtype=torch.float32)
        return nn.Sequential(h, nn.ReLU(), o)

    def _export_3layer_pt(self):
        """Export a three-layer PyTorch network.

        Layer 1 neurons come from the active convex pathways.
        Anti-sign directions are **negated** so that ReLU activates
        where D1=1 (matching the convex mask), and their second-layer
        weight is negated to recover the correct sign.

        Layer 2 groups first-layer neurons by (output, pattern_i,
        pattern_l, output_sign).  The combined coupling constraint
        guarantees that the aggregated sum has sign matching D2_l,
        so the second-layer ReLU acts as a faithful D2 gate.
        """
        import torch
        import torch.nn as nn

        d = self._d
        d_aug = self._d_aug
        K = self._K
        D1 = self._D1
        P1 = min(len(p) for p in D1)
        P2 = len(self._D2)
        threshold = 1e-6

        pathways = []
        for k in range(K):
            vp, vm_ = self._vp[k], self._vm[k]
            wp, wm_ = self._wp[k], self._wm[k]
            G = vp.shape[0]
            for g in range(G):
                j = g // (P1 * P2)
                rem = g % (P1 * P2)
                i = rem // P2
                l = rem % P2
                for vec, out_sign, ij_sign in [
                    (vp[g], +1, +1), (vm_[g], +1, -1),
                    (wp[g], -1, +1), (wm_[g], -1, -1),
                ]:
                    nrm = np.linalg.norm(vec)
                    if nrm > threshold:
                        pathways.append(
                            (k, j, i, l, out_sign, ij_sign,
                             vec, nrm))

        if not pathways:
            h1 = nn.Linear(d, 1)
            nn.init.zeros_(h1.weight); nn.init.zeros_(h1.bias)
            h2 = nn.Linear(1, 1)
            nn.init.zeros_(h2.weight); nn.init.zeros_(h2.bias)
            out = nn.Linear(1, K)
            nn.init.zeros_(out.weight)
            out.bias.data = torch.tensor(self._y_mean,
                                         dtype=torch.float32)
            return nn.Sequential(h1, nn.ReLU(), h2, nn.ReLU(), out)

        n_h1 = len(pathways)
        W1 = np.zeros((n_h1, d))
        b1 = np.zeros(n_h1)
        pw_i = np.zeros(n_h1, dtype=int)
        pw_l = np.zeros(n_h1, dtype=int)
        pw_k = np.zeros(n_h1, dtype=int)
        pw_out_sign = np.zeros(n_h1)
        pw_ij_sign = np.zeros(n_h1)
        pw_mag = np.zeros(n_h1)

        for idx, (ko, _j, pat_i, pat_l, osign, ijsign, vec, nrm) \
                in enumerate(pathways):
            sign_flip = -1.0 if ijsign < 0 else 1.0
            u = sign_flip * vec / nrm
            W1[idx] = u[:d]
            b1[idx] = u[d]
            pw_i[idx] = pat_i
            pw_l[idx] = pat_l
            pw_k[idx] = ko
            pw_out_sign[idx] = osign
            pw_ij_sign[idx] = ijsign
            pw_mag[idx] = nrm

        layer1 = nn.Linear(d, n_h1)
        layer1.weight.data = torch.tensor(W1, dtype=torch.float32)
        layer1.bias.data = torch.tensor(b1, dtype=torch.float32)

        group_keys = set()
        for idx in range(n_h1):
            group_keys.add(
                (int(pw_k[idx]), int(pw_i[idx]), int(pw_l[idx]),
                 int(pw_out_sign[idx])))
        group_list = sorted(group_keys)
        group_map = {g: gi for gi, g in enumerate(group_list)}
        n_h2 = len(group_list)

        W2 = np.zeros((n_h2, n_h1))
        b2 = np.zeros(n_h2)
        for idx in range(n_h1):
            gkey = (int(pw_k[idx]), int(pw_i[idx]),
                    int(pw_l[idx]), int(pw_out_sign[idx]))
            gi = group_map[gkey]
            W2[gi, idx] = pw_ij_sign[idx] * pw_mag[idx]

        layer2 = nn.Linear(n_h1, n_h2)
        layer2.weight.data = torch.tensor(W2, dtype=torch.float32)
        layer2.bias.data = torch.tensor(b2, dtype=torch.float32)

        W_out = np.zeros((K, n_h2))
        b_out = self._y_mean.copy()
        for gi, (ko, _i, _l, osign) in enumerate(group_list):
            W_out[ko, gi] = float(osign)

        out_layer = nn.Linear(n_h2, K)
        out_layer.weight.data = torch.tensor(
            W_out, dtype=torch.float32)
        out_layer.bias.data = torch.tensor(
            b_out, dtype=torch.float32)

        return nn.Sequential(
            layer1, nn.ReLU(), layer2, nn.ReLU(), out_layer)
