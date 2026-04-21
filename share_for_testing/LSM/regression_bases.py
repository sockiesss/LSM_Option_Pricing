import numpy as np
import numpy.polynomial.laguerre as lag
from abc import ABC, abstractmethod

class BaseRegression(ABC):
    """
    Abstract base class for polynomial regression bases.
    Handles the shared fitting and predicting logic.
    Works on arbitrary feature vectors, not just spot prices.
    Input X can be (n_samples, n_features).
    """
    def __init__(self, degree: int = 3):
        """
        Initialize regression with specified polynomial degree and regression parameters.  
        Bigger degree can cause multicollinearity.
        """
        self.degree = degree
        self.beta = None
    
    @abstractmethod
    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Build the regression design matrix used by fit/predict.
        Must be implemented by subclasses to define the specific basis functions.
        """
        pass

    def fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fits Y on X and returns beta
        """
        A = self.design_matrix(X)
        beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
        self.beta = beta
        return beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts on X using fitted beta
        """
        A = self.design_matrix(X)
        return A @ self.beta

    def fit_predict(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fit regression and return in-sample fitted values.
        Used for plain LSM (without LOO) so that design_matrix is built only once.
        """
        A = self.design_matrix(X)
        beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
        self.beta = beta
        return A @ beta
    
class LaguerrePolynomials(BaseRegression):
    """
    Works on arbitrary feature vectors, not just spot prices.
    Input X can be (n_samples, n_features).
    - For single-asset, generates Laguerre polynomials up to self.degree.
    - For multi-asset, expects X to be pre-computed features like cross-products (done by payoffs.py) and adds a constant term.
    """
    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            # Single-asset: Laguerre polynomial basis
            A = lag.lagvander(X, self.degree)
        else:
            # Multi-asset: features + constant term
            A = np.column_stack([np.ones(X.shape[0]), X])
        return A

class PowerPolynomials(BaseRegression):
    """
    Safe for negative domain inputs like raw option spreads.
    - For single-asset, generates power polynomials 1, x, x^2, ... up to self.degree.
    - For multi-asset, expects X to be pre-computed features like cross-products (done by payoffs.py) and adds a constant term.
    """
    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            # Single-asset: generate 1, X, X^2, ..., X^degree
            return np.vander(X, self.degree + 1, increasing=True)
        else:
            # Multi-asset: features + constant term
            return np.column_stack([np.ones(X.shape[0]), X])