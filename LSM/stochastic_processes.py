# Generate Geometric Brownian Motion paths for Stock Price
import warnings
import numpy as np

class GeometricBrownianMotion:
    """
    Simulates GBM paths for one or more correlated assets.
    Supports antithetic variance reduction.
    """
    def __init__(self, S0: np.ndarray, r: float, q: np.ndarray, sigma: np.ndarray,
                 correlation_matrix: np.ndarray = None):
        """
        Initialize GBM parameters (single or multi-asset).
        S0: spot price (scalar or array)
        r: risk-free rate
        q: dividend yield (scalar or array)
        sigma: volatility (scalar or array)
        correlation_matrix: correlation structure (default: identity)
        """
        S0 = np.atleast_1d(np.asarray(S0, dtype=np.float64))
        q = np.atleast_1d(np.asarray(q, dtype=np.float64))
        sigma = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
        
        self.n_assets = len(S0)
        self.S0 = S0
        self.r = r
        self.q = q
        self.sigma = sigma
        
        # Default to identity (independent assets) if not provided
        if correlation_matrix is None:
            self.correlation_matrix = np.eye(self.n_assets)
        else:
            self.correlation_matrix = correlation_matrix.astype(np.float64)
        
        # Cholesky decomposition for generating correlated normals
        self.L = np.linalg.cholesky(self.correlation_matrix) # Corr = L @ L.T

    def simulate(self, T: float, n_steps: int, n_paths: int, 
                 rng: np.random.Generator = None,
                 use_antithetic: bool = False,
                 simulation_times=None) -> tuple:
        """
        Simulate GBM paths.
        
        Args:
            T: time to maturity
            n_steps: time steps
            n_paths: number of paths (if use_antithetic=True, generates n_paths/2 pairs)
            rng: random number generator
            use_antithetic: if True, generate antithetic pairs for variance reduction
            simulation_times: optional 1-d array of monitoring times (including 0 and T);
                   overrides T and n_steps. Enables non-uniform grids for Bermudan/swing.
                           
        Returns:
            time_grid: 1-d array of shape (n_steps + 1,)
            paths: 2-d array of shape (n_paths, n_steps + 1) for single-asset,
                   3-d array of shape (n_paths, n_steps + 1, n_assets) for multi-asset
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if use_antithetic:
            if n_paths % 2 != 0:
                warnings.warn(f"n_paths={n_paths} is odd; the last path is dropped for antithetic pairing.")
            n_base_paths = n_paths // 2
        else:
            n_base_paths = n_paths

        if simulation_times is not None:
            time_grid = np.asarray(simulation_times, dtype=np.float64)
            if len(time_grid) < 2:
                raise ValueError("simulation_times must have at least 2 elements.")
            if time_grid[0] != 0.0:
                raise ValueError(f"simulation_times must start at 0, got {time_grid[0]}.")
            if np.any(np.diff(time_grid) <= 0):
                raise ValueError("simulation_times must be strictly increasing.")
            n_steps = len(time_grid) - 1
        else:
            n_steps = int(n_steps)
            time_grid = np.linspace(0, T, n_steps + 1)

        # Per-step dt: shape (n_steps,) — handles uniform and non-uniform grids uniformly
        dt = np.diff(time_grid)                                            # (n_steps,)
        drift     = (self.r - self.q - 0.5 * self.sigma**2) * dt[:, None] # (n_steps, n_assets)
        diffusion = self.sigma * np.sqrt(dt[:, None])                      # (n_steps, n_assets)

        paths_base = np.empty((n_base_paths, n_steps + 1, self.n_assets))
        paths_base[:, 0, :] = self.S0

        # Generate correlated normals
        Z_uncorr = rng.normal(size=(n_base_paths, n_steps, self.n_assets))
        Z = Z_uncorr @ self.L.T  # Corr(Z) = Cov(Z) = E(Z.T @ Z) = L @ L.T

        log_return = drift[None, :, :] + diffusion[None, :, :] * Z  # (n_base_paths, n_steps, n_assets)
        paths_base[:, 1:, :] = self.S0 * np.exp(np.cumsum(log_return, axis=1))

        # If using antithetic variables, create antithetic paths and combine
        if use_antithetic:
            paths_anti = np.empty((n_base_paths, n_steps + 1, self.n_assets))
            paths_anti[:, 0, :] = self.S0
            paths_anti[:, 1:, :] = self.S0 * np.exp(
                np.cumsum(drift[None, :, :] - diffusion[None, :, :] * Z, axis=1)
            )
            paths = np.vstack([paths_base, paths_anti])
        else:
            paths = paths_base
        
        # Squeeze single-asset case from (n, steps, 1) to (n, steps)
        if self.n_assets == 1:
            paths = paths.squeeze(axis=2)
        
        return time_grid, paths

class AmericanMaxCall:
    """
    Example:
        payoff = AmericanMaxCall(strike=100)
        intrinsic = payoff(asset_prices)  # shape (n_paths,)
    """
    def __init__(self, strike: float):
        self.strike = strike
    
    def __call__(self, asset_prices: np.ndarray) -> np.ndarray:
        return np.maximum(asset_prices - self.strike, 0)

class QuantoGBM:
    """
    Fixed-rate quanto stock process under the domestic pricing measure.

    Stock dynamics:
        dS_t = (r_for - q - rho_sfx * sigma_s * sigma_fx) S_t dt + sigma_s S_t dW_t

    Use with:
        vanilla payoff -> AmericanOption(...)
        quanto payoff  -> ScaledPayoff(AmericanOption(...), fx_fix)
    """
    def __init__(self, S0, r_dom, r_for, q, sigma_s, sigma_fx, rho_sfx):
        self.S0 = float(S0)
        self.r = float(r_dom)          # kept for compatibility / fallback
        self.r_dom = float(r_dom)
        self.r_for = float(r_for)
        self.q = float(q)
        self.sigma_s = float(sigma_s)
        self.sigma_fx = float(sigma_fx)
        self.rho_sfx = float(rho_sfx)

    def simulate(self, T, n_steps, n_paths, rng=None, use_antithetic=False):
        if rng is None:
            rng = np.random.default_rng()

        dt = T / n_steps
        time_grid = np.linspace(0.0, T, n_steps + 1)

        if use_antithetic:
            n_base = n_paths // 2
        else:
            n_base = n_paths

        Z = rng.normal(size=(n_base, n_steps))
        mu_q = self.r_for - self.q - self.rho_sfx * self.sigma_s * self.sigma_fx
        drift = (mu_q - 0.5 * self.sigma_s**2) * dt
        diff  = self.sigma_s * np.sqrt(dt)

        paths_base = np.zeros((n_base, n_steps + 1), dtype=np.float64)
        paths_base[:, 0] = self.S0

        for t in range(1, n_steps + 1):
            paths_base[:, t] = paths_base[:, t - 1] * np.exp(drift + diff * Z[:, t - 1])

        if use_antithetic:
            paths_anti = np.zeros((n_base, n_steps + 1), dtype=np.float64)
            paths_anti[:, 0] = self.S0
            for t in range(1, n_steps + 1):
                paths_anti[:, t] = paths_anti[:, t - 1] * np.exp(drift - diff * Z[:, t - 1])
            paths = np.vstack([paths_base, paths_anti])
        else:
            paths = paths_base

        return time_grid, paths


class QuantoStochasticRatesProcess:
    """
    Fixed-rate quanto with stochastic domestic / foreign short rates.

    State per path and time:
        state[..., 0] = stock S_t
        state[..., 1] = domestic short rate r_d(t)
        state[..., 2] = foreign short rate r_f(t)

    Dynamics:
        dS_t  = (r_f(t) - q - rho_sfx * sigma_s * sigma_fx) S_t dt + sigma_s S_t dW_s
        dr_d  = a_d (b_d - r_d) dt + sigma_d dW_d
        dr_f  = a_f (b_f - r_f) dt + sigma_f dW_f

    correlation_matrix is 3x3 for [W_s, W_d, W_f].
    """
    def __init__(
        self,
        S0,
        rd0,
        rf0,
        q,
        sigma_s,
        sigma_fx,
        rho_sfx,
        a_d,
        b_d,
        sigma_d,
        a_f,
        b_f,
        sigma_f,
        correlation_matrix=None
    ):
        self.S0 = float(S0)
        self.rd0 = float(rd0)
        self.rf0 = float(rf0)
        self.q = float(q)

        self.sigma_s = float(sigma_s)
        self.sigma_fx = float(sigma_fx)
        self.rho_sfx = float(rho_sfx)

        self.a_d = float(a_d)
        self.b_d = float(b_d)
        self.sigma_d = float(sigma_d)

        self.a_f = float(a_f)
        self.b_f = float(b_f)
        self.sigma_f = float(sigma_f)

        # kept for compatibility / fallback
        self.r = self.rd0

        if correlation_matrix is None:
            correlation_matrix = np.eye(3)

        self.correlation_matrix = np.asarray(correlation_matrix, dtype=np.float64)
        self.cholesky_factor = np.linalg.cholesky(self.correlation_matrix)

    def simulate(self, T, n_steps, n_paths, rng=None, use_antithetic=False):
        if rng is None:
            rng = np.random.default_rng()

        dt = T / n_steps
        sqdt = np.sqrt(dt)
        time_grid = np.linspace(0.0, T, n_steps + 1)

        if use_antithetic:
            n_base = n_paths // 2
        else:
            n_base = n_paths

        Z_uncorr = rng.normal(size=(n_base, n_steps, 3))
        Z = np.einsum("...j,ij->...i", Z_uncorr, self.cholesky_factor.T)

        states_base = np.zeros((n_base, n_steps + 1, 3), dtype=np.float64)
        states_base[:, 0, 0] = self.S0
        states_base[:, 0, 1] = self.rd0
        states_base[:, 0, 2] = self.rf0

        for t in range(1, n_steps + 1):
            s_prev  = states_base[:, t - 1, 0]
            rd_prev = states_base[:, t - 1, 1]
            rf_prev = states_base[:, t - 1, 2]

            z_s = Z[:, t - 1, 0]
            z_d = Z[:, t - 1, 1]
            z_f = Z[:, t - 1, 2]

            rd_new = rd_prev + self.a_d * (self.b_d - rd_prev) * dt + self.sigma_d * sqdt * z_d
            rf_new = rf_prev + self.a_f * (self.b_f - rf_prev) * dt + self.sigma_f * sqdt * z_f

            mu_q = rf_prev - self.q - self.rho_sfx * self.sigma_s * self.sigma_fx
            s_new = s_prev * np.exp((mu_q - 0.5 * self.sigma_s**2) * dt + self.sigma_s * sqdt * z_s)

            states_base[:, t, 0] = s_new
            states_base[:, t, 1] = rd_new
            states_base[:, t, 2] = rf_new

        if use_antithetic:
            states_anti = np.zeros((n_base, n_steps + 1, 3), dtype=np.float64)
            states_anti[:, 0, 0] = self.S0
            states_anti[:, 0, 1] = self.rd0
            states_anti[:, 0, 2] = self.rf0

            for t in range(1, n_steps + 1):
                s_prev  = states_anti[:, t - 1, 0]
                rd_prev = states_anti[:, t - 1, 1]
                rf_prev = states_anti[:, t - 1, 2]

                z_s = -Z[:, t - 1, 0]
                z_d = -Z[:, t - 1, 1]
                z_f = -Z[:, t - 1, 2]

                rd_new = rd_prev + self.a_d * (self.b_d - rd_prev) * dt + self.sigma_d * sqdt * z_d
                rf_new = rf_prev + self.a_f * (self.b_f - rf_prev) * dt + self.sigma_f * sqdt * z_f

                mu_q = rf_prev - self.q - self.rho_sfx * self.sigma_s * self.sigma_fx
                s_new = s_prev * np.exp((mu_q - 0.5 * self.sigma_s**2) * dt + self.sigma_s * sqdt * z_s)

                states_anti[:, t, 0] = s_new
                states_anti[:, t, 1] = rd_new
                states_anti[:, t, 2] = rf_new

            states = np.vstack([states_base, states_anti])
        else:
            states = states_base

        return time_grid, states

    def discount_step(self, paths: np.ndarray, t: int, dt: float) -> np.ndarray:
        """
        Pathwise discount factor from t to t+1 using domestic short rate at time t.
        """
        rd_t = paths[:, t, 1]
        return np.exp(-rd_t * dt)

