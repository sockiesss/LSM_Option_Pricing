import numpy as np
from scipy import stats
from LSM.stochastic_processes import GeometricBrownianMotion
from LSM.payoffs import VanillaPayoff
from LSM.regression_bases import LaguerrePolynomials
from LSM.algorithms import LeastSquaresMonteCarlo

def bsm_european_call(S0, K, T, r, q, sigma):
    """Helper function for BSM European Call closed-form price."""
    d1 = (np.log(S0/K) + (r - q + 0.5*(sigma**2))*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-q*T) * S0 * stats.norm.cdf(d1) - np.exp(-r*T) * K * stats.norm.cdf(d2)

def test_vanilla_put_pricing():
    """
    A basic sanity test to ensure the LSM pricer returns valid floats 
    and prices a standard American Put option.
    """
    gbm = GeometricBrownianMotion(S0=36.0, r=0.06, q=0.0, sigma=0.2)
    put_payoff = VanillaPayoff(strike=40.0, option_type="put")
    basis = LaguerrePolynomials(degree=3)

    lsm_engine = LeastSquaresMonteCarlo(
        process=gbm, payoff_function=put_payoff, basis_function=basis
    )

    price, std = lsm_engine.pricer(T=1.0, n_steps=50, n_paths=1000, rng=np.random.default_rng(42))

    assert price > 0.0, "Option price should be strictly positive."
    assert std >= 0.0, "Standard error cannot be negative."
    assert isinstance(price, float), "Price must be a float."

def test_american_call_no_dividend():
    """
    Sanity Check: An American Call option on a non-dividend paying stock (q=0) 
    should have the exact same price as a European Call option.
    """
    S0, K, T, r, q, sigma = 36.0, 40.0, 1.0, 0.06, 0.0, 0.2
    
    # 1. Price via LSM Engine
    gbm = GeometricBrownianMotion(S0=S0, r=r, q=q, sigma=sigma)
    call_payoff = VanillaPayoff(strike=K, option_type="call")
    basis = LaguerrePolynomials(degree=3)
    
    lsm_engine = LeastSquaresMonteCarlo(
        process=gbm, payoff_function=call_payoff, basis_function=basis
    )
    
    # Using 10,000 paths for a more accurate CI test
    lsm_price, _ = lsm_engine.pricer(T=T, n_steps=50, n_paths=10000, rng=np.random.default_rng(42))
    
    # 2. Price via Analytical BSM
    bsm_price = bsm_european_call(S0, K, T, r, q, sigma)
    
    # 3. Assert they are close (allowing for Monte Carlo variance, e.g., within 0.15)
    absolute_error = abs(lsm_price - bsm_price)
    assert absolute_error < 0.15, f"LSM price {lsm_price} deviated too far from BSM {bsm_price}"