import numpy as np

class SVM:
  def __init__(self,
        soft_margin: float = 1.0,
        n_iters: int = 1000,
        kernel: str = "linear",
        kernel_params: dict | None = None,
        tol: float = 1e-3,
        rng_seed: int | None = None):
    
    self.C = soft_margin
    self.n_iters = n_iters
    self.kernel = kernel
    self.kernel_params = kernel_params or {}
    self.tol = tol
    self.rng = np.random.default_rng(rng_seed)

    # learned params
    self.alpha = None
    self.b = 0.0
    self.X_sv = None
    self.y_sv = None
    self.w = None  # untuk linear

  def _kernel(self, X1, X2):
    if self.kernel == "linear":
      return X1 @ X2.T
    elif self.kernel == "poly":
      degree = self.kernel_params.get("degree", 3)
      coef0 = self.kernel_params.get("coef0", 1.0)
      return (X1 @ X2.T + coef0) ** degree
    elif self.kernel == "rbf":
      gamma = self.kernel_params.get("gamma", 1.0 / X1.shape[1])
      X1_sq = np.sum(X1**2, axis=1)[:, None]
      X2_sq = np.sum(X2**2, axis=1)[None, :]
      sq_dists = X1_sq + X2_sq - 2 * (X1 @ X2.T)
      return np.exp(-gamma * np.clip(sq_dists, 0.0, None))
    else:
      raise ValueError("Unknown kernel")

  def fit(self, X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    y = np.where(y <= 0, -1.0, 1.0)  # pastikan {-1,+1}

    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    b = 0.0

    K = self._kernel(X, X)

    for _ in range(self.n_iters):
      num_changed = 0
      for i in range(n_samples):
        f_i = np.sum(alpha * y * K[:, i]) + b
        E_i = f_i - y[i]

        # KKT violation
        if (y[i]*E_i < -self.tol and alpha[i] < self.C) or (y[i]*E_i > self.tol and alpha[i] > 0):
            # pick j != i (random)
            j = self.rng.integers(0, n_samples - 1)
            if j >= i:
                j += 1

            f_j = np.sum(alpha * y * K[:, j]) + b
            E_j = f_j - y[j]

            ai_old, aj_old = alpha[i], alpha[j]

            # bounds
            if y[i] != y[j]:
                L = max(0.0, aj_old - ai_old)
                H = min(self.C, self.C + aj_old - ai_old)
            else:
                L = max(0.0, ai_old + aj_old - self.C)
                H = min(self.C, ai_old + aj_old)

            if L == H:
                continue

            # eta
            eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
            if eta <= 0:
                continue

            # update aj
            aj_new = aj_old - y[j] * (E_i - E_j) / eta
            aj_new = np.clip(aj_new, L, H)
            if abs(aj_new - aj_old) < 1e-5:
                continue

            # update ai
            ai_new = ai_old + y[i]*y[j]*(aj_old - aj_new)

            # bias updates
            b1 = b - E_i - y[i]*(ai_new - ai_old)*K[i, i] - y[j]*(aj_new - aj_old)*K[i, j]
            b2 = b - E_j - y[i]*(ai_new - ai_old)*K[i, j] - y[j]*(aj_new - aj_old)*K[j, j]

            # commit
            alpha[i], alpha[j] = ai_new, aj_new
            if 0 < ai_new < self.C:
                b = b1
            elif 0 < aj_new < self.C:
                b = b2
            else:
                b = 0.5 * (b1 + b2)

            num_changed += 1

      # (opsional) early break kalau satu epoch tidak ada perubahan
      if num_changed == 0:
        break

    # simpan SV
    sv_mask = alpha > 1e-5
    self.alpha = alpha[sv_mask]
    self.X_sv = X[sv_mask]
    self.y_sv = y[sv_mask]
    self.b = b

    # bonus: konstruksi w untuk linear agar prediksi cepat
    if self.kernel == "linear" and self.alpha.size > 0:
        self.w = (self.alpha[:, None] * self.y_sv[:, None] * self.X_sv).sum(axis=0)
    else:
        self.w = None

    return self

  def decision_function(self, X):
    X = np.asarray(X, dtype=float)
    if self.kernel == "linear" and self.w is not None:
        return X @ self.w + self.b
    if self.alpha is None or self.alpha.size == 0:
        return np.full(X.shape[0], self.b)
    K = self._kernel(X, self.X_sv)
    return (K * (self.alpha * self.y_sv)).sum(axis=1) + self.b

  def predict(self, X):
    scores = self.decision_function(X)
    yhat = np.sign(scores)
    # mapping sign(0)=+1 agar tidak 0
    yhat[yhat == 0] = 1.0
    return yhat
