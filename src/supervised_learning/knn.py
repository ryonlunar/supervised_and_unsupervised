import numpy as np
import pandas as pd

class KNN:
  def __init__ (self, k=3, distance='euclidean', p=None):
    # constructor
    self.k = k
    self.distance = distance
    self.X_train = None
    self.y_train = None

    if self.distance == 'minkowski':
      if p == None:
        raise ValueError("Parameter p must be specified for Minkowski distance")
      self.distance_function = self._minkowski_distance
    elif self.distance == 'euclidean':
      self.distance_function = self._euclidean_distance
    elif self.distance == 'manhattan':
      self.distance_function = self._manhattan_distance
    else:
      raise ValueError("Unknown distance metric")
    
  def fit(self, X_train, y_train):
    self.X_train = np.array(X_train)
    self.y_train = np.array(y_train)

  def predict(self, X_test):
    if self.X_train is None or self.y_train is None:
      raise ValueError("Model has not been trained yet. Call fit() before predict().")
    X_test = np.array(X_test)
    return np.array([self._predict_single(x) for x in X_test])

  def _predict_single(self, x):
    distances = [self.distance_function(x, x_train) for x_train in self.X_train]
    k_nearest_indices = np.argsort(distances)[:self.k]
    k_nearest_labels = self.y_train[k_nearest_indices]
    most_common_label = pd.Series(k_nearest_labels).mode()[0] # katanya agak lama
    return most_common_label

  def _euclidean_distance(self, a, b):
    return np.sqrt(np.sum((a - b) ** 2))

  def _manhattan_distance(self, a, b):
    return np.sum(np.abs(a - b))

  def _minkowski_distance(self, a, b):
    return np.sum(np.abs(a - b) ** self.p) ** (1/self.p)