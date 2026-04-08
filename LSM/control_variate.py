
import numpy as np
from scipy.stats import norm

def bs_european_price(S0, K: float, r: float, q: float,
                      sigma: float, T, option_type: str):
    """
    Black-Scholes closed-form price for European call/put.
    S0 and T accept scalars or arrays; returns float or ndarray accordingly.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    S0 = np.asarray(S0, dtype=float)
    T  = np.asarray(T,  dtype=float)
    scalar = S0.ndim == 0 and T.ndim == 0
    S0, T = np.atleast_1d(S0), np.atleast_1d(T)

    # T <= 0: intrinsic value
    if option_type == "call":
        at_expiry = np.maximum(S0 - K, 0.0)
    else:
        at_expiry = np.maximum(K - S0, 0.0)

    # sigma = 0: deterministic discounted forward
    Tpos = np.maximum(T, 0.0)
    forward_intrinsic = S0 * np.exp(-q * Tpos) - K * np.exp(-r * Tpos)
    if option_type == "call":
        sigma_zero = np.maximum(forward_intrinsic, 0.0)
    else:
        sigma_zero = np.maximum(-forward_intrinsic, 0.0)

    # Normal BS formula; safe_T avoids 0/0 on T=0 paths
    safe_T = np.where(T > 0, T, 1.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * safe_T) / (sigma * np.sqrt(safe_T))
    d2 = d1 - sigma * np.sqrt(safe_T)
    if option_type == "call":
        bs = S0 * np.exp(-q * safe_T) * norm.cdf(d1) - K * np.exp(-r * safe_T) * norm.cdf(d2)
    else:
        bs = K * np.exp(-r * safe_T) * norm.cdf(-d2) - S0 * np.exp(-q * safe_T) * norm.cdf(-d1)

    result = np.where(T <= 0, at_expiry, np.where(sigma <= 0, sigma_zero, bs))
    return float(result) if scalar else result


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
