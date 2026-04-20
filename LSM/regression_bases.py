import numpy as np
import numpy.polynomial.laguerre as lag

class LaguerrePolynomials:
    """
    Works on arbitrary feature vectors, not just spot prices.
    Input X can be (n_samples, n_features).
    """
    def __init__(self, degree: int = 3):
        self.degree = degree
        self.beta = None

    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Build the regression design matrix used by fit/predict.
        """
        if X.ndim == 1:
            # Single-asset: Laguerre polynomial basis
            A = lag.lagvander(X, self.degree)
        else:
            # Multi-asset: features + constant term
            A = np.column_stack([np.ones(X.shape[0]), X])
        return A

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