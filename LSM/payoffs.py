import numpy as np

class AmericanOption:
    def __init__(self, strike: float, option_type: str = "put"):
        """
        Initializes the American option's payoff parameters
        strike: strike price
        option_type: "call" or "put"
        """
        self.option_type = option_type
        self.strike = strike

    def __call__(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Returns the intrinsic value of the American option
        spot_prices: array of spot prices at any given time step
        """
        if self.option_type == "put":
            return np.maximum(self.strike - spot_prices, 0.0)
        else:
            return np.maximum(spot_prices - self.strike, 0.0)