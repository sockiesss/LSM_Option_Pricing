import unittest
import numpy as np
from scipy import stats
from LSM.stochastic_processes import GeometricBrownianMotion
from LSM.payoffs import VanillaPayoff
from LSM.regression_bases import LaguerrePolynomials
from LSM.algorithms import LeastSquaresMonteCarlo
from LSM.control_variate import bs_european_price


def bsm_european_call(S0, K, T, r, q, sigma):
    """Helper function for BSM European Call closed-form price."""
    d1 = (np.log(S0/K) + (r - q + 0.5*(sigma**2))*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-q*T) * S0 * stats.norm.cdf(d1) - np.exp(-r*T) * K * stats.norm.cdf(d2)

class TestLSMEngine(unittest.TestCase):
    def setUp(self):
        """This runs before every single test to set up a clean baseline."""
        self.rng = np.random.default_rng(42)
        self.base_process = GeometricBrownianMotion(S0=100, r=0.05, q=0.0, sigma=0.2)
        self.base_payoff = VanillaPayoff(strike=100, option_type="put")
        self.base_basis = LaguerrePolynomials(degree=3)
        self.base_engine = LeastSquaresMonteCarlo(self.base_process, self.base_payoff, self.base_basis)
    
    def test_vanilla_put_pricing(self):
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

    def test_american_call_no_dividend(self):
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
    
        # 3. Assert they are close (allowing for Monte Carlo variance)
        absolute_error = abs(lsm_price - bsm_price)
        assert absolute_error < 0.15, f"LSM price {lsm_price} deviated too far from BSM {bsm_price}"


    def test_deep_otm_price_is_zero(self):
        """
        Deep OTM option should have exactly zero value.
        """
        # Spot is 100, Strike is 10. Impossible to exercise.
        payoff = VanillaPayoff(strike=10, option_type="put") 
        engine = LeastSquaresMonteCarlo(self.base_process, payoff, self.base_basis)
        
        price, _ = engine.pricer(T=1.0, n_steps=50, n_paths=1000, rng=self.rng)
        self.assertAlmostEqual(price, 0.0, places=4, msg=f"Deep OTM Put should be 0, got {price}")

    def test_zero_volatility_converges_to_intrinsic(self):
        """
        With ~0 volatility, American put price equals discounted intrinsic if exercised.
        """
        process = GeometricBrownianMotion(S0=100, r=0.05, q=0.0, sigma=0.0001) 
        payoff = VanillaPayoff(strike=110, option_type="put") # ITM Put
        engine = LeastSquaresMonteCarlo(process, payoff, self.base_basis)
        
        price, _ = engine.pricer(T=1.0, n_steps=50, n_paths=1000, rng=self.rng)
        expected_intrinsic = 10.0 # (110 - 100)
        
        self.assertGreaterEqual(price, expected_intrinsic, "Zero-vol ITM American Put must be >= immediate intrinsic")

    def test_american_greater_than_european(self):
        """
        Financial Benchmark: American price must be >= European closed form.
        """
        american_price, _ = self.base_engine.pricer(T=1.0, n_steps=50, n_paths=5000, rng=self.rng)
        
        euro_price = bs_european_price(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="put")
        
        self.assertGreaterEqual(american_price, euro_price, f"American ({american_price:.4f}) must be >= European ({euro_price:.4f})")

    def test_cv_logic_variance_reduction(self):
        """Requirement: Control Variate should reduce stderr, not change price significantly."""
        # Note: Use a standard GBM as CV is disabled for Quanto in your logic
        gbm = GeometricBrownianMotion(S0=100, r=0.05, q=0.0, sigma=0.2)
        payoff = VanillaPayoff(strike=100, option_type="put")
        engine = LeastSquaresMonteCarlo(gbm, payoff, LaguerrePolynomials(3))
    
        p_no_cv, std_no_cv = engine.pricer(T=1.0, n_steps=50, n_paths=2000, control_variate=None, rng=np.random.default_rng(42))
        p_cv, std_cv = engine.pricer(T=1.0, n_steps=50, n_paths=2000, control_variate='european_at_maturity', rng=np.random.default_rng(42))
    
        assert std_cv < std_no_cv, "Control Variate failed to reduce variance."
        assert abs(p_cv - p_no_cv) < 3 * std_no_cv, "CV price drifted too far from base price."

if __name__ == '__main__':
    unittest.main()