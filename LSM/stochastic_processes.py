# Generate Geometric Brownian Motion paths for Stock Price
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
                 use_antithetic: bool = False) -> tuple:
        """
        Simulate GBM paths.
        
        Args:
            T: time to maturity
            n_steps: time steps
            n_paths: number of paths (if use_antithetic=True, generates n_paths/2 pairs)
            rng: random number generator
            use_antithetic: if True, generate antithetic pairs for variance reduction
                           
        Returns:
            time_grid: 1-d array of shape (n_steps + 1,)
            paths: 2-d array of shape (n_paths, n_steps + 1) for single-asset,
                   3-d array of shape (n_paths, n_steps + 1, n_assets) for multi-asset
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if use_antithetic:
            n_base_paths = n_paths // 2
        else:
            n_base_paths = n_paths
        
        n_steps = int(n_steps)
        dt = T / n_steps
        time_grid = np.linspace(0, T, n_steps + 1)
        paths_base = np.zeros((n_base_paths, n_steps + 1, self.n_assets))
        paths_base[:, 0, :] = self.S0

        # Generate correlated normals
        Z_uncorr = rng.normal(size=(n_base_paths, n_steps, self.n_assets))
        Z = Z_uncorr @ self.L.T  # Corr(Z) = Cov(Z) = E(Z.T @ Z) = L @ L.T
        drift = (self.r - self.q - 0.5 * (self.sigma**2)) * dt
        diffusion = self.sigma * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            paths_base[:, t, :] = paths_base[:, t-1, :] * np.exp(drift[None, :] + diffusion[None, :] * Z[:, t-1, :])
        
        # If using antithetic variables, create antithetic paths and combine
        if use_antithetic:
            paths_anti = np.zeros((n_base_paths, n_steps + 1, self.n_assets))
            paths_anti[:, 0, :] = self.S0
            for t in range(1, n_steps + 1):
                paths_anti[:, t, :] = paths_anti[:, t-1, :] * np.exp(drift[None, :] - diffusion[None, :] * Z[:, t-1, :])
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
