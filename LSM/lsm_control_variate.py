
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


def lsm_with_control_variate(lsm_engine,
                             T: float,
                             n_steps: int,
                             n_paths: int,
                             rng: np.random.Generator = None,
                             use_antithetic: bool = False,
                             create_features=None,
                             cache: bool = False) -> tuple[float, float, float]:
    """
    use existing LSM setup, and apply a European-option control variate using Black-Scholes closed form.
    Only apply to one asset.

    """
    process = lsm_engine.process
    payoff_function = lsm_engine.payoff_function

    if create_features is not None:
        raise ValueError("This control variate wrapper is currently for single-asset options only.")

    # Reproduce the LSM logic externally so original files stay untouched.
    time_grid, paths = process.simulate(
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        rng=rng,
        use_antithetic=use_antithetic
    )

    if paths.ndim != 2:
        raise ValueError("Only single-asset paths are supported by this wrapper.")

    dsc_cashflow = payoff_function(paths[:, -1])

    cashflow_matrix = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    cashflow_matrix[:, -1] = dsc_cashflow

    for t in range(n_steps - 2, -1, -1):
        df = np.exp(-process.r * (time_grid[t + 1] - time_grid[t]))
        dsc_cashflow *= df

        immediate_payoff = payoff_function(paths[:, t])

        itm_mask = immediate_payoff > 0
        if not np.any(itm_mask):
            continue

        continuation = np.zeros_like(immediate_payoff, dtype=np.float64)

        strike = payoff_function.strike
        lsm_engine.basis_function.fit(paths[itm_mask, t] / strike, dsc_cashflow[itm_mask])
        continuation[itm_mask] = lsm_engine.basis_function.predict(paths[itm_mask, t] / strike)

        exercise_mask = immediate_payoff > continuation
        cashflow_matrix[exercise_mask, t] = immediate_payoff[exercise_mask]
        cashflow_matrix[exercise_mask, t + 1:] = 0.0
        dsc_cashflow = np.where(exercise_mask, immediate_payoff, dsc_cashflow)

    x_samples = dsc_cashflow

    S0 = float(np.atleast_1d(process.S0)[0])
    r = float(process.r)
    q = float(np.atleast_1d(process.q)[0])
    sigma = float(np.atleast_1d(process.sigma)[0])
    K = float(payoff_function.strike)
    option_type = payoff_function.option_type.lower()

    y_samples = european_discounted_payoff(paths[:, -1], K, r, T, option_type)
    y_expectation = bs_european_price(S0, K, r, q, sigma, T, option_type)

    price_cv, stderr_cv, beta = apply_control_variate(x_samples, y_samples, y_expectation)

    if cache:
        lsm_engine._cached_cashflow = cashflow_matrix
        lsm_engine._cached_cv_beta = beta
        lsm_engine._cached_euro_closed = y_expectation
        lsm_engine._cached_paths = paths

    return price_cv, stderr_cv
