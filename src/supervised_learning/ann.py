import numpy as np
from typing import List, Literal, Optional, Dict

# activation functions
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def d_sigmoid(a): return a * (1 - a)

def relu(z): return np.maximum(0.0, z)
def d_relu(z): return (z > 0).astype(float)

def linear(z): return z
def d_linear(_): return 1.0

def softmax(z):
  # stabil secara numerik
  z_shift = z - np.max(z, axis=1, keepdims=True)
  e = np.exp(z_shift)
  return e / (e.sum(axis=1, keepdims=True) + 1e-12)

ACTS = {
  "sigmoid": (sigmoid, d_sigmoid),
  "relu":    (relu, d_relu),
  "linear":  (linear, d_linear),
  "softmax": (softmax, None),   # grad ditangani bareng cross-entropy
}

# init functions
def init_weights(fan_in, fan_out, method="xavier", rng=None):
  rng = rng or np.random.default_rng(42)
  if method == "xavier":
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    W = rng.uniform(-limit, limit, size=(fan_in, fan_out))
  elif method == "he":
    std = np.sqrt(2.0 / fan_in)
    W = rng.normal(0.0, std, size=(fan_in, fan_out))
  elif method == "zeros":
    W = np.zeros((fan_in, fan_out))
  else:
    raise ValueError("Unknown init method")
  b = np.zeros((1, fan_out))
  return W, b

# loss functions
def mse_loss(y_true, y_pred):
  # y_true: (N, C or 1), y_pred: sama
  diff = y_pred - y_true
  return 0.5 * np.mean(np.sum(diff * diff, axis=1))

def d_mse(y_true, y_pred):
  # dL/dA = (y_pred - y_true) / N, tapi N dicicil di GD batch
  return (y_pred - y_true)

def cross_entropy_loss(y_true_onehot, y_pred_prob):
  # y_true_onehot dan y_pred_prob: (N, C)
  eps = 1e-12
  return -np.mean(np.sum(y_true_onehot * np.log(y_pred_prob + eps), axis=1))

class ANN:
  def __init__(self,
                layer_sizes: List[int],           # contoh: [n_in, h1, h2, n_out]
                activations: List[str],           # contoh: ["relu", "relu", "softmax"]
                init: str = "xavier",
                loss: Literal["mse", "cross_entropy"] = "mse",
                reg_type: Literal["none", "l1", "l2"] = "none",
                reg_lambda: float = 0.0,
                learning_rate: float = 1e-2,
                epochs: int = 1000,
                batch_size: int = 32,
                seed: Optional[int] = 42,
                verbose: bool = False
                ):
    assert len(layer_sizes) >= 2, "Minimal input & output."
    assert len(activations) == len(layer_sizes) - 1, "Aktivasi per-layer (kecuali input)."
    self.L = len(layer_sizes) - 1
    self.sizes = layer_sizes
    self.acts = activations
    self.init = init
    self.loss_name = loss
    self.reg_type = reg_type
    self.reg_lambda = reg_lambda
    self.lr = learning_rate
    self.epochs = epochs
    self.bs = batch_size
    self.rng = np.random.default_rng(seed)
    self.verbose = verbose

    # parameter
    self.W: List[np.ndarray] = []
    self.b: List[np.ndarray] = []
    for l in range(self.L):
      fan_in, fan_out = layer_sizes[l], layer_sizes[l+1]
      Wl, bl = init_weights(fan_in, fan_out, method=init, rng=self.rng)
      self.W.append(Wl)
      self.b.append(bl)

  def _forward(self, X: np.ndarray):
    A = X
    cache = {"Z": [], "A": [A]}  # simpan semua Z, A
    for l in range(self.L):
      Z = A @ self.W[l] + self.b[l]
      act_name = self.acts[l]
      f, df = ACTS[act_name]
      if act_name == "softmax":
        A = f(Z)
      elif act_name == "sigmoid":
        A = f(Z)
      else:
        A = f(Z)
      cache["Z"].append(Z)
      cache["A"].append(A)
    return A, cache

  def _reg_loss_and_grad(self):
    if self.reg_type == "none" or self.reg_lambda <= 0.0:
      return 0.0, [0]*self.L
    if self.reg_type == "l2":
      reg_loss = 0.5 * self.reg_lambda * sum((W*W).sum() for W in self.W)
      reg_grads = [self.reg_lambda * W for W in self.W]
    elif self.reg_type == "l1":
      reg_loss = self.reg_lambda * sum(np.abs(W).sum() for W in self.W)
      reg_grads = [self.reg_lambda * np.sign(W) for W in self.W]
    else:
      raise ValueError("Unknown reg_type")
    return reg_loss, reg_grads

  def _backward(self, cache, y_true):
    grads_W = [None]*self.L
    grads_b = [None]*self.L

    A_out = cache["A"][-1]           # prediksi terakhir
    N = y_true.shape[0]

    # grad w.r.t. A_out (dA)
    if self.loss_name == "mse":
      dA = d_mse(y_true, A_out)    # (N, C)
    elif self.loss_name == "cross_entropy":
      # asumsi output activation = softmax; dL/dZ = (A_out - y_true)
      dA = (A_out - y_true)
    else:
      raise ValueError("Unknown loss")

    # Backprop per layer (dari belakang)
    for l in reversed(range(self.L)):
      Zl = cache["Z"][l]
      Al_prev = cache["A"][l]      # input ke layer l
      act_name = self.acts[l]
      if act_name == "softmax" and self.loss_name == "cross_entropy":
        # langsung gunakan dZ = dA (karena turunan softmax+CE = A - y)
        dZ = dA
      else:
        # dZ = dA * f'(Z)
        f, df = ACTS[act_name]
        if act_name == "sigmoid":
          # pakai a = sigmoid(z) agar stabil
          a = cache["A"][l+1]
          dZ = dA * df(a)
        else:
          dZ = dA * df(Zl)

      grads_W[l] = (Al_prev.T @ dZ) / N
      grads_b[l] = np.sum(dZ, axis=0, keepdims=True) / N

      # untuk layer sebelumnya
      if l > 0:
        dA = dZ @ self.W[l].T

    return grads_W, grads_b

  def _update(self, grads_W, grads_b, reg_grads):
      for l in range(self.L):
        gW = grads_W[l] + (0 if reg_grads == 0 else reg_grads[l])
        self.W[l] -= self.lr * gW
        self.b[l] -= self.lr * grads_b[l]

  def fit(self, X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N = X.shape[0]
    steps_per_epoch = max(1, (N + self.bs - 1) // self.bs)

    for epoch in range(self.epochs):
      # shuffle
      idx = self.rng.permutation(N)
      Xs, ys = X[idx], y[idx]

      epoch_loss = 0.0
      for step in range(steps_per_epoch):
        s = step * self.bs
        e = min(N, s + self.bs)
        Xb, yb = Xs[s:e], ys[s:e]

        # forward
        y_pred, cache = self._forward(Xb)

        # loss
        if self.loss_name == "mse":
            data_loss = mse_loss(yb, y_pred)
        else:
            data_loss = cross_entropy_loss(yb, y_pred)

        reg_loss, reg_grads = self._reg_loss_and_grad()
        batch_loss = data_loss + reg_loss / max(1, steps_per_epoch)  # disebar per-batch

        # backward
        grads_W, grads_b = self._backward(cache, yb)

        # update
        self._update(grads_W, grads_b, reg_grads)

        epoch_loss += batch_loss

      if self.verbose:
        if (epoch+1) % max(1, self.epochs//10) == 0:
          print(f"epoch {epoch+1}/{self.epochs} - loss {epoch_loss/steps_per_epoch:.4f}")

    return self

  def predict_proba(self, X):
    X = np.asarray(X, dtype=float)
    y_pred, _ = self._forward(X)
    return y_pred

  def predict(self, X):
    proba = self.predict_proba(X)
    if self.acts[-1] == "softmax":
        return np.argmax(proba, axis=1)
    return proba
