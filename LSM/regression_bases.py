import numpy as np
import numpy.polynomial.laguerre as lag

class LaguerrePolynomials:
    """
    Works on arbitrary feature vectors, not just spot prices.
    Input X can be (n_samples, n_features).
    """
    def __init__(self, degree: int = 3):
        """
        Initializes the degree parameter of the Laguerre Polynomial
        degree: max order, 3 by default
        beta: fitted beta 1-d np.array, None by default, only valid after fitting
        """
        self.degree = degree
        self.beta = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fits Y on X and returns beta
        
        Args:
            X: shape (n_samples,) for single-asset, or (n_samples, n_features) for multi-asset
            Y: shape (n_samples,)
        
        Returns:
            beta: fitted coefficients
        """
        if X.ndim == 1:
            # Single-asset: use Laguerre polynomial basis
            A = lag.lagvander(X, self.degree)  # (n_samples, degree + 1)
        else:
            # Multi-asset: use features with constant term
            A = np.column_stack([np.ones(X.shape[0]), X])  # (n_samples, n_features + 1)
        
        beta, *_ = np.linalg.lstsq(A, Y, rcond=None) # Don't store A (large matrix)
        self.beta = beta
        return beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts on X using fitted beta
        
        Args:
            X: shape (n_samples,) for single-asset, or (n_samples, n_features) for multi-asset
        
        Returns:
            predictions: shape (n_samples,)
        """
        if X.ndim == 1:
            # Single-asset: use Laguerre polynomial basis
            A = lag.lagvander(X, self.degree)
        else:
            # Multi-asset: use features with constant term
            A = np.column_stack([np.ones(X.shape[0]), X])
        
        return A @ self.beta