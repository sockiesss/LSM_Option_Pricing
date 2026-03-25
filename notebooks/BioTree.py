from scipy.stats import norm
import numpy as np

def Binomial_price_Am (S: np.array, K: float, r: float, q: float, sigma: float, T: float, N: int, option_type: str='call') -> float:
    """Compute the price of an American put option using binomial model
    S: float: initial price
    K: float: strike price
    r: float: risk-free rate
    sigma: float: volatility
    T: float: maturity
    N: int: time steps
    """
    if option_type.lower() == "call":
        cp = 1
    elif option_type.lower() == "put":
        cp = -1
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    dt = T / N
    
    # u and d in CRR model, and the risk neutral measure
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    
    # stock prices for the last period
    idx = np.arange(0, N+1)
    ST = S * np.power(u, idx) * np.power(d, N-idx)
    
    # Payoff for the last period
    payoff = np.maximum(cp * (ST-K), 0)
    
    # Iteration backward through the tree
    for period in range(N):
        idx = np.arange (0, N - period)
        ST = S * np.power (u, idx) * np.power (d, N-period-1-idx) # use the S0 to calculate St instead of ST
        payoff = np.maximum(cp * (ST-K), np.exp(-r*dt)* (p * payoff[1:] + (1-p)* payoff[:-1])) # compare the price backward and the current payoff
        
    # The array is of size 1 but we return the value only so that the returned value is of type float
    return payoff[0]
