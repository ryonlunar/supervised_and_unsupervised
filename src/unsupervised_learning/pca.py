import numpy as np

class PCA:
    def __init__(self, n_components):
      self.n_components = n_components
      self.components_ = None
      self.explained_variance_ratio_ = None
      self.mean_ = None

    def fit(self, X):
      X = np.array(X)
      self.mean_ = np.mean(X, axis=0)
      X_centered = X - self.mean_

      cov_matrix = np.cov(X_centered, rowvar=False)
      
      eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

      sorted_idx = np.argsort(eigenvalues)[::-1]
      eigenvalues = eigenvalues[sorted_idx]
      eigenvectors = eigenvectors[:, sorted_idx]

      self.components_ = eigenvectors[:, :self.n_components]

      self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)

      return self

    def transform(self, X):
      # Proyeksikan data ke komponen utama
      X_centered = X - self.mean_
      return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
      self.fit(X)
      return self.transform(X)
