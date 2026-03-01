
import numpy as np
import numpy.polynomial.laguerre as lag

class LaguerrePolynomials:
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
        """
        A = lag.lagvander(X, self.degree) # 2-d np.ndarray, size=(N, degree + 1)
        beta, *_ = np.linalg.lstsq(A, Y, rcond=None)  # Don't store A (large matrix)
        self.beta = beta
        return beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts on X using fitted beta; can be a new X for testing
        """
        if self.beta is None: # if not self.beta may raise error since self.beta is an array
            raise ValueError("Please fit first before predicting!")
        A = lag.lagvander(X, self.degree) # Compute A again in case predicted on new X
        return A @ self.beta