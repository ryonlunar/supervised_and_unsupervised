import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any


class Node:
  def __init__ (self,*, feature: int = None, threshold: float = None, left: 'Node' = None, right: 'Node' = None, value: float = None):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value
    
  def is_leaf(self):
    return self.value is not None


class DecisionTreeRegressor:
  def __init__ (self, 
                max_depth = 6,
                min_samples_split = 10,
                min_samples_leaf = 5,
                min_impurity_decrease = 1e-7
                ):
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_samples_leaf = min_samples_leaf
    self.min_impurity_decrease = min_impurity_decrease
    self.root = None
    
  def _mse(self, y):
    if y.size == 0:
      return 0
    else:
      return np.mean((y - np.mean(y)) ** 2)
  
  def _best_split(self, X, y):
    n, d = X.shape
    parent_mse = self._mse(y)
    best_feat, best_thr, best_gain = None, None, 0
    
    for j in range(d):
      xs = X[:,j]
      order = np.argsort(xs)
      xs_sorted = xs[order]
      y_sorted = y[order]
      
      y_cum = np.cumsum(y_sorted)
      y2_cum = np.cumsum(y_sorted ** 2)
      
      for i in range(self.min_samples_leaf, n - self.min_samples_leaf):
        if xs_sorted[i] == xs_sorted[i-1]:
          continue
        
        left_count = i
        right_count = n - i
        
        left_sum = y_cum[i-1]
        right_sum = y_cum[-1] - left_sum
        
        left_sum2 = y2_cum[i-1]
        right_sum2 = y2_cum[-1] - left_sum2
        
        left_mse = (left_sum2 - (left_sum ** 2) / left_count) / left_count
        right_mse = (right_sum2 - (right_sum ** 2) / right_count) / right_count
        
        weighted_mse = (left_count * left_mse + right_count * right_mse) / n
        gain = parent_mse - weighted_mse
        
        if gain > best_gain:
          best_gain = gain
          best_feat = j
          best_thr = (xs_sorted[i] + xs_sorted[i-1]) / 2
          
    return best_feat, best_thr, best_gain
      
      
  def _build(self, X, y, depth):
    # kalau tidak ada sample
    if y.size == 0:
        return Node(value=0.0)

    # kondisi berhenti
    if (depth >= self.max_depth) or (y.size < self.min_samples_split):
        return Node(value=float(y.mean()))

    feat, thr, gain = self._best_split(X, y)
    if feat is None or gain < self.min_impurity_decrease:
        return Node(value=float(y.mean()))

    left_mask = X[:, feat] <= thr
    right_mask = ~left_mask

    # guard min_samples_leaf atau split jelek
    if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
        return Node(value=float(y.mean()))

    left = self._build(X[left_mask], y[left_mask], depth+1)
    right = self._build(X[right_mask], y[right_mask], depth+1)

    return Node(feature=feat, threshold=float(thr), left=left, right=right)

  def fit (self, X, y):
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    self.root = self._build(X, y, depth=0)
    return self
  
  def _predict_single(self, x, node: Node):
    if node.is_leaf():
      return node.value

    if x[node.feature] <= node.threshold:
      return self._predict_single(x, node.left)
    else:
      return self._predict_single(x, node.right)
  
  def predict(self, X):
    if self.root is None:
      raise ValueError("Model has not been trained yet. Call fit() before predict().")
    X = np.asarray(X)
    return np.array([self._predict_single(x, self.root) for x in X])
  
  def export_rules(self, feature_names=None, node=None, depth=0):
    """Return list of string rules (preorder)."""
    if node is None:
      node = self.root
    lines = []
    indent = "  " * depth
    if node.is_leaf():
      lines.append(f"{indent}-> value = {node.value:.6f}")
    else:
      fname = f"x[{node.feature}]" if feature_names is None else feature_names[node.feature]
      lines.append(f"{indent}if {fname} <= {node.threshold:.6f}:")
      lines += self.export_rules(feature_names, node.left, depth+1)
      lines.append(f"{indent}else:  # {fname} > {node.threshold:.6f}")
      lines += self.export_rules(feature_names, node.right, depth+1)
    return lines
