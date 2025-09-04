import numpy as np

def compute_distance(p1, p2, metric="euclidean", p=3):
  """Hitung jarak antara dua titik dengan metric tertentu"""
  diff = np.abs(p1 - p2)
  if metric == "euclidean":
      return np.sqrt(np.sum(diff ** 2))
  elif metric == "manhattan":
      return np.sum(diff)
  elif metric == "minkowski":
      return np.sum(diff ** p) ** (1/p)
  else:
      raise ValueError("Metric harus 'euclidean', 'manhattan', atau 'minkowski'")

class DBSCAN:
  def __init__(self, eps=0.5, min_samples=5, metric="euclidean", p=3):
    self.eps = eps
    self.min_samples = min_samples
    self.metric = metric
    self.p = p
    self.labels_ = None

  def fit_predict(self, X):
    X = np.array(X)
    n = len(X)
    self.labels_ = np.full(n, -1)  # -1 artinya noise
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
          continue
        visited[i] = True

        # Cari tetangga titik i
        neighbors = self._region_query(X, i)
        if len(neighbors) < self.min_samples:
          self.labels_[i] = -1  # noise
        else:
          # Buat cluster baru
          self._expand_cluster(X, i, neighbors, cluster_id, visited)
          cluster_id += 1
    return self

  def _region_query(self, X, idx):
    neighbors = []
    for j in range(len(X)):
      d = compute_distance(X[idx], X[j], metric=self.metric, p=self.p)
      if d <= self.eps:
        neighbors.append(j)
    return neighbors

  def _expand_cluster(self, X, idx, neighbors, cluster_id, visited):
    self.labels_[idx] = cluster_id
    i = 0
    while i < len(neighbors):
      n_idx = neighbors[i]
      if not visited[n_idx]:
        visited[n_idx] = True
        n_neighbors = self._region_query(X, n_idx)
        if len(n_neighbors) >= self.min_samples:
          neighbors.extend(n_neighbors)  # tambahkan tetangga baru
      if self.labels_[n_idx] == -1:
        self.labels_[n_idx] = cluster_id
      i += 1
