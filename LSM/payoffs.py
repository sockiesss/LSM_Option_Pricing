import numpy as np
from numpy.polynomial import hermite

class VanillaPayoff:
    def __init__(self, strike: float, option_type: str = "put"):
        """
        Initializes the American option's payoff parameters
        strike: strike price
        option_type: "call" or "put"
        """
        self.option_type = option_type.lower()
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

class MaxCallFeatures:
    """
    Features for American call on maximum of N assets.
    Transforms N asset prices to 3N+3 manually crafted features.
    See Longstaff-Schwartz (2001) pp. 141-142 for the five-asset case.
    """
    
    def __init__(self, strike: float):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        self.strike = strike
    
    def __call__(self, asset_prices: np.ndarray) -> np.ndarray:
        """
        Args:
            asset_prices: (n_paths, N) where N >= 2
        
        Returns:
            features: (n_paths, 3N+3)
            
        Features: constant, 5 Hermite in max, raw 2nd-Nth, squares of 2nd-Nth, 
        adjacent products, all-product
        """
        n_paths, n_assets = asset_prices.shape
        n_features = 3 * n_assets + 3
        features = np.zeros((n_paths, n_features))
        
        # Sort all paths at once
        sorted_prices = np.sort(asset_prices, axis=1)  # (n_paths, n_assets)
        s_max = sorted_prices[:, -1]  # (n_paths,)
        s_rest = sorted_prices[:, :-1]  # (n_paths, n_assets-1)
        
        idx = 0
        
        # 5 Hermite polynomials in max (constant is handled in the BaseRegression class)
        hermite_max = hermite.hermvander(s_max, 5)[:, 1:]  # (n_paths, 5)
        features[:, idx:idx+5] = hermite_max
        idx += 5
        
        # Raw states 2nd through Nth
        features[:, idx:idx+n_assets-1] = s_rest
        idx += n_assets - 1
        
        # Squares 2nd through Nth
        features[:, idx:idx+n_assets-1] = s_rest ** 2
        idx += n_assets - 1
        
        # Adjacent products
        features[:, idx:idx+n_assets-1] = sorted_prices[:, :-1] * sorted_prices[:, 1:]
        idx += n_assets - 1
        
        # Product of all
        features[:, idx] = np.prod(sorted_prices, axis=1)
        
        return features


class AmericanMaxCall:
    """American call on maximum of N assets."""
    
    def __init__(self, strike: float):
        self.strike = strike
    
    def __call__(self, asset_prices: np.ndarray) -> np.ndarray:
        """
        Args:
            asset_prices: (n_paths, N) 
        
        Returns:
            payoff: (n_paths,) = max(max(prices) - K, 0)
        """
        max_price = np.max(asset_prices, axis=1)
        return np.maximum(max_price - self.strike, 0.0)

class SwingSpread:
    def __init__(self, option_type: str = "call"):
        """
        Calculates the raw spread (difference between spot price and contract price).
        option_type: str, call/put
        """
        self.option_type = option_type.lower()

    def __call__(self, spot_prices: np.ndarray, contract_price: float) -> np.ndarray:
        """
        spot_prices: array of spot prices at the current time step
        contract_price: the specific strike/contract price at the current time step
        """
        if self.option_type == "put":
            return contract_price - spot_prices
        else:
            return spot_prices - contract_price

# -----------------------------------------------------------------------------
# Quanto payoff helpers
# -----------------------------------------------------------------------------
class ScaledPayoff:
    """
    Generic wrapper that multiplies any existing payoff by a fixed scalar.

    In the quanto setting, the scalar is usually the fixed FX conversion rate.
    For clearer quanto code, prefer FixedFXQuantoPayoff below.

    Example:
        vanilla_put = VanillaPayoff(strike=100, option_type="put")
        scaled_put = ScaledPayoff(vanilla_put, scale=1.2)
    """
    def __init__(self, base_payoff, scale: float):
        self.base_payoff = base_payoff
        self.scale = float(scale)

        # expose strike if underlying payoff has it, so existing normalization still works
        if hasattr(base_payoff, "strike"):
            self.strike = base_payoff.strike

    def __call__(self, state):
        return self.scale * self.base_payoff(state)


class FixedFXQuantoPayoff(ScaledPayoff):
    """
    Fixed-FX quanto payoff wrapper.

    This is the clearer name for the common quanto case where the underlying
    payoff is paid in a foreign asset/currency and converted using a fixed FX
    rate.

    Mathematically:
        quanto_payoff(state) = fx_fix * base_payoff(state)

    Example:
        vanilla_put = VanillaPayoff(strike=100, option_type="put")
        quanto_put = FixedFXQuantoPayoff(vanilla_put, fx_fix=1.2)
    """
    def __init__(self, base_payoff, fx_fix: float):
        super().__init__(base_payoff=base_payoff, scale=fx_fix)
        self.fx_fix = float(fx_fix)




class StateColumnPayoff:
    """
    Apply an existing 1D payoff to one selected column of a multi-dimensional state.

    Example:
        base_put = AmericanOption(strike=K, option_type="put")
        stock_put = StateColumnPayoff(base_put, column=0)   # use stock only
        quanto_put = ScaledPayoff(stock_put, scale=fx_fix)
    """
    def __init__(self, base_payoff, column: int = 0):
        self.base_payoff = base_payoff
        self.column = int(column)

        if hasattr(base_payoff, "strike"):
            self.strike = base_payoff.strike

    def __call__(self, state):
        if isinstance(state, np.ndarray) and state.ndim == 2:
            return self.base_payoff(state[:, self.column])
        return self.base_payoff(state)




class QuantoRateFeatures:
    """
    Feature map for stochastic-rate quanto state:
        state[:, 0] = stock
        state[:, 1] = domestic short rate
        state[:, 2] = foreign short rate
    """
    def __init__(self, strike: float):
        self.strike = float(strike)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        s  = state[:, 0] / self.strike
        rd = state[:, 1]
        rf = state[:, 2]

        return np.column_stack([
            s, rd, rf,
            s**2, rd**2, rf**2,
            s * rd, s * rf, rd * rf
        ])
