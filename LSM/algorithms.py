import numpy as np

class LeastSquaresMonteCarlo:
    """
    LSM Pricing engine based on Longstaff, Schwartz (2001)
    """
    def __init__(self, process, payoff_function, basis_function):
        self.process = process
        self.payoff_function = payoff_function
        self.basis_function = basis_function

    def pricer(self, T: float, n_steps: int, n_paths: int, 
               rng: np.random.Generator = None) -> float:
        """
        Monte Carlo simulation with regressions; uses backward induction 
        and compares continuation value and intrinsic value to decide whether
        to exercise
        Args:
            T: time to maturity
            n_steps: number of discretized steps
            n_paths: number of paths to simulate
            rng: numpy random number generator, empty by default
        Returns:
            option price
        """
        time_grid, paths = self.process.simulate(T, n_steps, n_paths, rng)
        
        # TODO: cashflow matrix, backward induction, etc. See paper pp. 115-120, p. 122
        cashflow = self.payoff_function(paths[:, -1])

        # for t in range(n_steps - 1, 0, -1):
        # Evaluate the immediate payoff at the current step, compare it to the predicted
        # continuation value, and update cashflows if early exercise is optimal

        return 0.0