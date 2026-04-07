
import numpy as np
from scipy.stats import norm

def bs_european_price(S0: float, K: float, r: float, q: float,
                      sigma: float, T: float, option_type: str) -> float:
    """
    Black-Scholes closed-form price for European call/put
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    # Handle maturity = 0
    if T <= 0:
        if option_type == "call":
            return max(S0 - K, 0.0)
        else:
            return max(K - S0, 0.0)

    # Optional: handle sigma = 0 separately
    if sigma <= 0:
        forward_intrinsic = S0 * np.exp(-q * T) - K * np.exp(-r * T)
        if option_type == "call":
            return max(forward_intrinsic, 0.0)
        else:
            return max(-forward_intrinsic, 0.0)

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)


def european_discounted_payoff(ST: np.ndarray, K: float, r: float, T: float,
                               option_type: str) -> np.ndarray:
    """
    Pathwise discounted price for European call/put
    """
    option_type = option_type.lower()
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return np.exp(-r * T) * payoff


def apply_control_variate(x_samples: np.ndarray,
                          y_samples: np.ndarray,
                          y_expectation: float) -> tuple[float, float, float]:
    """
    control variate estimator: X_cv = X - beta * (Y - E[Y])

    Returns: 
    price_cv : float
        Control-variate price estimate
    stderr_cv : float
        Standard error of the control-variate estimator
    beta : float
        Estimated optimal beta = Cov(X,Y)/Var(Y)
    """
    x_samples = np.asarray(x_samples, dtype=float)
    y_samples = np.asarray(y_samples, dtype=float)

    if x_samples.shape != y_samples.shape:
        raise ValueError("x_samples and y_samples must have the same shape")

    n = x_samples.size
    if n == 0:
        raise ValueError("Samples must be non-empty")

    if n == 1:
        beta = 0.0
        cv_samples = x_samples.copy()
        return float(cv_samples[0]), 0.0, beta

    var_y = np.var(y_samples, ddof=1) # variance of asset
    beta = 0.0 if var_y == 0 else np.cov(x_samples, y_samples, ddof=1)[0, 1] / var_y # best beta

    cv_samples = x_samples - beta * (y_samples - y_expectation) # X_cv = X - beta * (Y - E[Y])
    price_cv = float(np.mean(cv_samples))
    stderr_cv = float(np.std(cv_samples, ddof=1) / np.sqrt(n))
    return price_cv, stderr_cv, float(beta)
