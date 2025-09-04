import numpy as np
import pandas as pd
from itertools import combinations_with_replacement

class PolynomialRegression:
  def __init__(self, degree=2, regularization=None, lambda_param=0.01, iterations=1000,verbose=False, learning_rate=0.01, optimization='gd'):
    self.X_train = None
    self.y_train = None
    self.degree = degree
    self.regularization = regularization
    self.learning_rate = learning_rate
    self.lambda_param = lambda_param
    self.iterations = iterations
    self.weights = None
    self.loss_history = []
    self.verbose = verbose
    self.optimization = optimization

  def _transform_features(self, X):
    X = np.atleast_2d(X)
    n_samples, n_features = X.shape
    X_poly = np.ones((n_samples, 1))

    for d in range(1, self.degree + 1):
      feature_indices_combinations = combinations_with_replacement(range(n_features), d)
      for combo in feature_indices_combinations:
        poly_feature = np.prod(X[:, combo], axis=1, keepdims=True)
        X_poly = np.c_[X_poly, poly_feature]
    return X_poly
  
  def fit(self, X_train, y_train):
    X_prepared = np.array(X_train)
    y_prepared = np.array(y_train).reshape(-1, 1)
    
    X_poly = self._transform_features(X_prepared)
    if self.optimization == 'gd':
      self._fit_gradient_descent(X_poly, y_prepared)
    elif self.optimization == 'newton':
      self._fit_newton_method(X_poly, y_prepared)
    else:
      raise ValueError("Unknown optimization method. Supported: 'gd' (gradient descent)")

  def _fit_gradient_descent(self, X_train, y_train):
    n_samples, n_features = X_train.shape
    # init bobot
    self.weights = np.zeros((n_features, 1))
    
    for i in range(self.iterations):
      y_pred = X_train @ self.weights
      error = y_pred - y_train
      loss = np.mean(np.power(error, 2)) # mean squared error
      self.loss_history.append(loss)

      gradient = (2 / n_samples) * (X_train.T @ error) # turunan dari mse
      
      if self.verbose and i % 50 == 0:
        print(f"Iteration {i}, Loss: {loss}")
      
      if self.regularization is not None:
        reg_term_grad = np.copy(self.weights)
        reg_term_grad[0] = 0
        if self.regularization == 'l2':
            gradient += self.lambda_param * reg_term_grad
        elif self.regularization == 'l1':
            gradient += self.lambda_param * np.sign(reg_term_grad)

      self.weights -= self.learning_rate * gradient
      
  def _fit_newton_method(self, X_train, y_train):
    
    if self.regularization == 'l1':
      raise ValueError("Newton's method does not support L1 regularization.")
    
    n_features = X_train.shape[1]

    identity_matrix = np.identity(n_features)
    identity_matrix[0, 0] = 0
    
    try:
      XTX = X_train.T @ X_train
      if self.regularization == 'l2':
        XTX += self.lambda_param * identity_matrix
      
      XTy = X_train.T @ y_train
      self.weights = np.linalg.solve(XTX, XTy) # XTX w = XTy
      
    except np.linalg.LinAlgError as e:
      raise ValueError("Matrix inversion failed. Check if the matrix is singular or ill-conditioned.") from e

  def predict(self, X):
    if self.weights is None:
      raise ValueError("Model has not been trained yet. Call fit() before predict().")
    X = np.atleast_2d(X)
    X_poly = self._transform_features(X)
    y_pred = X_poly @ self.weights
    return y_pred.flatten()