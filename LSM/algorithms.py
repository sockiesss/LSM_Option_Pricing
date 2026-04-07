import numpy as np
from control_variate import *

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
               control_variate: str = 'european_at_exercise',
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
        
        # dataframe for tracking stopping time / stopping spot (control variate part)
        exercise_time = np.full(n_paths, T, dtype=np.float64)
        exercise_spot = paths[:, -1].copy()
        exercise_index = np.full(n_paths, n_steps, dtype=int)

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
            
            # record stopping rule
            if control_variate == 'european_at_exercise':
                exercise_time[exercise_mask] = time_grid[t]
                exercise_spot[exercise_mask] = paths[exercise_mask, t]
                exercise_index[exercise_mask] = t
           
            # Update discounted cashflow for the next iteration
            dsc_cashflow = np.where(exercise_mask, immediate_payoff, dsc_cashflow)

        price = np.mean(dsc_cashflow)
        if n_paths > 1:
            stderr = np.std(dsc_cashflow, ddof=1) / np.sqrt(n_paths)
        else:
            stderr = 0.0

        # Optional European control variate
        if control_variate is not None:
            if create_features is not None:
                raise ValueError("European control variate is currently only supported for single-asset options.")

            if paths.ndim != 2:
                raise ValueError("European control variate is currently only supported for single-asset paths.")

            required_attrs = ["strike", "option_type"]
            for attr in required_attrs:
                if not hasattr(self.payoff_function, attr):
                    raise ValueError(f"payoff_function must have attribute '{attr}' for European control variate.")
            
            S0 = float(np.atleast_1d(self.process.S0)[0])
            r = float(self.process.r)
            q = float(np.atleast_1d(self.process.q)[0])
            sigma = float(np.atleast_1d(self.process.sigma)[0])
            K = float(self.payoff_function.strike)
            option_type = self.payoff_function.option_type.lower()

            x_samples = dsc_cashflow
            
            if control_variate == "european_at_maturity":
                # Y_i = discounted European payoff at maturity
                y_samples = european_discounted_payoff(paths[:, -1], K, r, T, option_type)
                y_expectation = bs_european_price(S0, K, r, q, sigma, T, option_type)
            elif control_variate == "european_at_exercise":
                # Y_i = exp(-r tau_i) * C(S_tau_i, T - tau_i)
                # E[Y] = C(S0, T)
                remaining_T = np.maximum(T - exercise_time, 0.0)

                # calculate european price for every exercising time
                euro_value_at_tau = np.array([
                    bs_european_price(
                        S0=exercise_spot[i],
                        K=K,
                        r=r,
                        q=q,
                        sigma=sigma,
                        T=remaining_T[i],
                        option_type=option_type
                    )
                    for i in range(n_paths)
                ])
                
                y_samples = np.exp(-r * exercise_time) * euro_value_at_tau
                y_expectation = bs_european_price(S0, K, r, q, sigma, T, option_type)
            
            else:
                raise ValueError("control_variate must be one of: None, 'european_at_maturity', 'european_at_exercise'")
            
            price, stderr, beta = apply_control_variate(x_samples, y_samples, y_expectation)

            if cache:
                self._cached_cv_beta = beta
                self._cached_euro_closed = y_expectation
                self._cached_exercise_time = exercise_time
                self._cached_exercise_spot = exercise_spot
                self._cached_exercise_index = exercise_index
                self._cached_paths = paths
                self._cached_cv_samples = y_samples

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