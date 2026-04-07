from scipy.stats import norm
import numpy as np
from payoffs import VanillaPayoff

class BinomialTreeEngine:
    """
    Binomial Tree pricing engine for vanilla European / American options
    using the Cox-Ross-Rubinstein (CRR) model.
    """

    def __init__(self, payoff_function: VanillaPayoff):
        self.payoff_function = payoff_function

    def pricer(self,S0: float,r: float,q: float,sigma: float,T: float,n_steps: int,
        american: bool = True,
        cache: bool = False
    ) -> float:
        """
        Price a vanilla option using the CRR binomial tree.

        Args:
            S0: initial spot price
            r: risk-free interest rate
            q: continuous dividend yield
            sigma: volatility
            T: time to maturity
            n_steps: number of time steps in the tree
            american: if True, allow early exercise; otherwise price as European
            cache: if True, store terminal stock prices / last backward values

        Returns:
            price: option price
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be a positive integer")
        if T <= 0:
            return float(self.payoff_function(np.array([S0]))[0])
        if sigma < 0:
            raise ValueError("sigma must be non-negative")

        dt = T / n_steps

        # CRR up/down factors and risk-neutral probability
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-r * dt)
        growth = np.exp((r - q) * dt)
        p = (growth - d) / (u - d)

        # Terminal stock prices at maturity
        idx = np.arange(n_steps + 1)
        ST = S0 * (u ** idx) * (d ** (n_steps - idx))

        # Terminal payoff
        option_values = self.payoff_function(ST)

        # Backward induction
        for step in range(n_steps - 1, -1, -1):
            idx = np.arange(step + 1)
            # Continuation value
            continuation = disc * (
                p * option_values[1:] + (1.0 - p) * option_values[:-1]
            )

            if american:
                # Stock prices at current step
                St = S0 * (u ** idx) * (d ** (step - idx))
                exercise = self.payoff_function(St)
                option_values = np.maximum(exercise, continuation)
            else:
                option_values = continuation #can add other types of options

        price = float(option_values[0])

        if cache:
            self._cached_terminal_spots = ST
            self._cached_price_tree_last_layer = option_values.copy()

        return price