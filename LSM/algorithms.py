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
               rng: np.random.Generator = None,
               use_antithetic: bool = False,
               create_features=None,
               cache: bool = False) -> tuple:
        """
        Monte Carlo simulation with regressions; uses backward induction 
        and compares continuation value and intrinsic value to decide whether
        to exercise.
        
        Args:
            T: time to maturity
            n_steps: number of discretized steps
            n_paths: number of paths to simulate (or 2*base paths if use_antithetic=True)
            rng: numpy random number generator, empty by default
            use_antithetic: if True, use antithetic variable variance reduction
            create_features: function to create additional features for regression
            cache: if True, cache the cash flow matrix with exercise cash flows
            
        Returns:
            (price, stderr): option price estimate and standard error
            
        Call get_cashflow() to retrieve cached cashflow matrix (only if cache=True).
        """
        time_grid, paths = self.process.simulate(T, n_steps, n_paths, rng, use_antithetic=use_antithetic)
        
        # Initializations: See Longstaff-Schwartz paper pp. 115-120, p. 122
        # 1. Array of discounted ex-post realized cash flow (for one-step regression)
        dsc_cashflow = self.payoff_function(paths[:, -1])
        
        # 2. Cash flow matrix: shape (n_paths, n_steps+1) to align with time_grid and paths
        cashflow_matrix = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        cashflow_matrix[:, -1] = dsc_cashflow  # Copy terminal payoffs

        for t in range(n_steps - 2, -1, -1):
            # Discount the cash flow from the next step back to the current step
            df = np.exp(-self.process.r * (time_grid[t+1] - time_grid[t]))
            dsc_cashflow *= df
            
            # Immediate payoff at current step
            immediate_payoff = self.payoff_function(paths[:, t])

            # Fit continuation value on in-the-money paths
            itm_mask = immediate_payoff > 0
            if not np.any(itm_mask):
                continue
            
            # Normalize state by strike for numerical stability. See Longstaff-Schwartz paper p. 143.
            # For multi-asset, create features first and don't normalize.
            continuation = np.zeros_like(immediate_payoff, dtype=np.float64)
            if create_features is None and hasattr(self.payoff_function, 'strike'):
                # Single asset with strike: normalize by strike
                strike = self.payoff_function.strike
                self.basis_function.fit(paths[itm_mask, t] / strike, dsc_cashflow[itm_mask])
                continuation[itm_mask] = self.basis_function.predict(paths[itm_mask, t] / strike)
            elif create_features is None:
                # Single asset without strike
                self.basis_function.fit(paths[itm_mask, t], dsc_cashflow[itm_mask])
                continuation[itm_mask] = self.basis_function.predict(paths[itm_mask, t])
            else:
                # Multi-asset
                features = create_features(paths[itm_mask, t, :])
                self.basis_function.fit(features, dsc_cashflow[itm_mask])
                continuation[itm_mask] = self.basis_function.predict(features) 
            
            # Update cashflow table
            # Note: For non-exercised paths, the cashflow is not updated.
            # 1. Exercise if immediate payoff is greater than continuation value
            exercise_mask = immediate_payoff > continuation
            cashflow_matrix[exercise_mask, t] = immediate_payoff[exercise_mask]
            # 2. For exercised paths, set future cashflow to zero since the option is exercised
            cashflow_matrix[exercise_mask, t+1:] = 0.0
           
            # Update discounted cashflow for the next iteration
            dsc_cashflow = np.where(exercise_mask, immediate_payoff, dsc_cashflow)
        
        price = np.mean(dsc_cashflow)
        if n_paths > 1:
            stderr = np.std(dsc_cashflow, ddof=1) / np.sqrt(n_paths)
        else:
            stderr = 0.0

        # Cache cash flow matrix only if requested
        if cache:
            self._cached_cashflow = cashflow_matrix
        
        return price, stderr
    
    def get_cashflow(self) -> np.ndarray:
        """
        Retrieve the cached cashflow matrix from the last pricer() call.
        
        Returns:
            cashflow_matrix: shape (n_paths, n_steps+1)
              Entry [i, t] = payoff if path i exercises at time t, else 0.
              
        Raises:
            RuntimeError: if cache=False was used in last pricer() call.
        """
        if not hasattr(self, '_cached_cashflow'):
            raise RuntimeError("Call pricer(cache=True) first before retrieving exercise decisions.")
        return self._cached_cashflow