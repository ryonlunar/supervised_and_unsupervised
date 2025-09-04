import pandas as pd
import numpy as np

class KMeans:
  def __init__(self, k=3,
              max_iter=100,
              init_method = "random",
              random_state = 42,
              ):
    self.k = k
    self.max_iter = max_iter
    self.init_method = init_method
    self.random_state = random_state
    self.centroids = None
    self.labels_ = None

  def _init_centroids_random(self, X):
    np.random.seed(self.random_state)
    random_indices = np.random.permutation(X.shape[0])[:self.k]
    return X[random_indices]

  def _init_centroids_kmeanspp(self, X):
    rng = np.random.default_rng(self.random_state)
    centroids = []

    centroids.append(X[rng.integers(0, X.shape[0])])
    
    for _ in range(1, self.k):
      distances = np.min(
                np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2) ** 2,
                axis=1
            )
      probs = distances / np.sum(distances)
      new_centroid = X[rng.choice(len(X), p=probs)]
      centroids.append(new_centroid)

    return np.array(centroids)

  def fit(self, X):
    X = np.array(X)
    
    # init centroid
    if self.init_method == "random":
      self.centroids = self._init_centroids_random(X)
    elif self.init_method == "kmeans++":
      self.centroids = self._init_centroids_kmeanspp(X)
    else:
      raise ValueError("Unknown initialization method")
    
    for _ in range(self.max_iter):
      distance = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
      labels = np.argmin(distance, axis=1)
      
      new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else self.centroids[i]
                for i in range(self.k)
            ])

      if np.allclose(new_centroids, self.centroids):
        break

      self.centroids = new_centroids

    self.labels_ = labels
    
    return self

  def predict(self, X):
    X = np.array(X)
    distance = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
    labels = np.argmin(distance, axis=1)
    return labels