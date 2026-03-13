# Generate Geometric Brownian Motion paths for Stock Price
import numpy as np

class GeometricBrownianMotion:
    def __init__(self, S0: float, r: float, q: float, sigma: float):
        """
        Initializes the paramters of the GBM SDE specific to the asset
        S0: spot price
        r: riskfree rate
        q: continuous dividend
        sigma: volatility
        """
        self.S0 = S0
        self.r =  r
        self.q = q
        self.sigma = sigma

    def simulate(self, T: float, n_steps: int, n_paths: int, 
                 rng: np.random.Generator = None) -> tuple:
        """
        Args:
            T: time to maturity
            n_steps: number of discretized steps
            n_paths: number of paths to simulate
            rng: numpy random number generator, empty by default
        Returns:
            time_grid: 1-d np.ndarray
            paths: a 2-d np.ndarray of GBM paths, size = (n_paths, n_steps + 1)
        """
        if rng is None:
            rng = np.random.default_rng()
            
        dt = T / n_steps
        time_grid = np.linspace(0, T, n_steps + 1)
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0

        Z = rng.normal(size = (n_paths, n_steps))

        drift = (self.r - self.q - 0.5 * (self.sigma**2)) * dt
        diffusion = self.sigma * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                drift + diffusion * Z[:, t-1]
            )
        
        return time_grid, paths

