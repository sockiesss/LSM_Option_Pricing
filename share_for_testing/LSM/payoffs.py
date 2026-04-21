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
    Transforms N asset prices → 3N+4 manually crafted features.
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
            features: (n_paths, 3N+4)
            
        Features: constant, 5 Hermite in max, raw 2nd-Nth, squares of 2nd-Nth, 
        adjacent products, all-product
        """
        n_paths, n_assets = asset_prices.shape
        n_features = 3 * n_assets + 4
        features = np.zeros((n_paths, n_features))
        
        # Sort all paths at once
        sorted_prices = np.sort(asset_prices, axis=1)  # (n_paths, n_assets)
        s_max = sorted_prices[:, -1]  # (n_paths,)
        s_rest = sorted_prices[:, :-1]  # (n_paths, n_assets-1)
        
        idx = 0
        
        # Constant + 5 Hermite polynomials in max
        hermite_max = hermite.hermvander(s_max, 5)  # (n_paths, 6)
        features[:, idx:idx+6] = hermite_max
        idx += 6
        
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


class ScaledPayoff:
    """
    Wrap an existing payoff and multiply it by a fixed scalar.
    Use this to turn the existing vanilla payoff into a fixed-rate quanto payoff.

    Example:
        vanilla_put = AmericanOption(strike=100, option_type="put")
        quanto_put  = ScaledPayoff(vanilla_put, scale=fx_fix)
    """
    def __init__(self, base_payoff, scale: float):
        self.base_payoff = base_payoff
        self.scale = float(scale)

        # expose strike if underlying payoff has it, so existing normalization still works
        if hasattr(base_payoff, "strike"):
            self.strike = base_payoff.strike

    def __call__(self, state):
        return self.scale * self.base_payoff(state)


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



class AmericanBestOfCall:
    """
    Bermudan/American call on the best of n assets:
        payoff = max(max_j S_j - K, 0)
    """
    def __init__(self, strike: float):
        self.strike = float(strike)

    def __call__(self, asset_prices: np.ndarray) -> np.ndarray:
        max_price = np.max(asset_prices, axis=1)
        return np.maximum(max_price - self.strike, 0.0)


class BestOfCallFeatures:
    """
    Feature map for best-of call on 2+ assets.

    For the 2-asset case in the LOOLSM paper, a good feature set is:
        1, payoff, s1, s2, s1^2, s1*s2, s2^2, s1^3, s1^2*s2, s1*s2^2, s2^3
    We return everything except the intercept because your regression basis
    already adds the constant internally for multi-dimensional features.
    """
    def __init__(self, strike: float):
        self.strike = float(strike)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        s1 = state[:, 0] / self.strike
        s2 = state[:, 1] / self.strike
        payoff = np.maximum(np.maximum(s1, s2) - 1.0, 0.0)

        return np.column_stack([
            payoff,
            s1, s2,
            s1**2, s1 * s2, s2**2,
            s1**3, (s1**2) * s2, s1 * (s2**2), s2**3
        ])



class AmericanBasketCall:
    """
    Bermudan/American call on the arithmetic average basket:
        payoff = max(avg(S_1,...,S_n) - K, 0)
    """
    def __init__(self, strike: float):
        self.strike = float(strike)

    def __call__(self, asset_prices: np.ndarray) -> np.ndarray:
        basket = np.mean(asset_prices, axis=1)
        return np.maximum(basket - self.strike, 0.0)


class BasketCallFeatures:
    """
    Feature map for arithmetic basket call on 4 assets.

    Inspired by the LOOLSM paper:
        1, payoff, s_j, s_j^2, s_i*s_j
    Again, we omit the intercept because the basis class adds it.
    """
    def __init__(self, strike: float):
        self.strike = float(strike)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        s = state / self.strike
        payoff = np.maximum(np.mean(s, axis=1) - 1.0, 0.0)

        cols = [payoff]

        # linear terms
        for j in range(s.shape[1]):
            cols.append(s[:, j])

        # squares
        for j in range(s.shape[1]):
            cols.append(s[:, j] ** 2)

        # pairwise cross terms
        for i in range(s.shape[1]):
            for j in range(i + 1, s.shape[1]):
                cols.append(s[:, i] * s[:, j])

        return np.column_stack(cols)
