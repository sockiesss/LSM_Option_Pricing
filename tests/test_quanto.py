import unittest
import numpy as np
from LSM.algorithms import LeastSquaresMonteCarlo
from LSM.payoffs import VanillaPayoff, QuantoPut, QuantoRateFeatures, CompositePut
from LSM.regression_bases import LaguerrePolynomials
from LSM.stochastic_processes import GeometricBrownianMotion, QuantoGBM, QuantoStochasticRatesProcess

class TestQuantoLSM(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for quanto tests."""
        self.S0 = 100
        self.K = 100
        self.T = 1.0
        self.r_dom = 0.05
        self.r_for = 0.02
        self.q = 0.0
        self.sigma_s = 0.2
        self.sigma_fx = 0.1
        self.rho_sfx = 0.3

    def test_quanto_martingale_property(self):
        """
        Martingale Test: E[ S_T * exp(-int(r_d dt)) ] must equal theoretical forward
        under the domestic risk-neutral measure, NOT S0!
        """
        S0 = 100.0
        quanto = QuantoStochasticRatesProcess(
            S0=S0, rd0=0.05, rf0=0.02, q=0.0, 
            sigma_s=0.2, sigma_fx=0.1, rho_sfx=0.3,
            a_d=1.0, b_d=0.05, sigma_d=0.01, 
            a_f=1.0, b_f=0.02, sigma_f=0.01
        )
        
        # Generate paths
        time_grid, paths = quanto.simulate(T=1.0, n_steps=100, n_paths=10000, rng=np.random.default_rng(42))
        
        # paths[:, :, 0] is the Stock
        dsc_S_T = paths[:, -1, 0].copy()
        
        # Apply pathwise discounting step-by-step
        dt = 1.0 / 100 
        for t in range(100 - 1, -1, -1):
            dsc_S_T *= quanto.discount_step(paths, t, dt)
            
        expected_forward = np.mean(dsc_S_T)
        
        # The expected discounted forward for a Quanto is S0 * exp((rf - rd - q - rho * sigma_s * sigma_fx) * T)
        drift_adjustment = 0.02 - 0.05 - 0.0 - (0.3 * 0.2 * 0.1)
        theoretical_expectation = S0 * np.exp(drift_adjustment * 1.0)
        
        self.assertAlmostEqual(expected_forward, theoretical_expectation, delta=1.5, 
            msg=f"Quanto drift is wrong! Expected ~{theoretical_expectation:.2f}, got {expected_forward:.2f}")

    def test_quanto_collapses_to_gbm(self):
        """
        Structural Test: A Quanto process with 0 rate volatility and 0 correlation 
        must price identically to a standard 1D GBM.
        """
        rng_1 = np.random.default_rng(42)
        rng_2 = np.random.default_rng(42)
        
        S0, K, T = 100.0, 100.0, 1.0
        r_dom = 0.05
        sigma_s = 0.2
        
        # 1. Standard 1D GBM Baseline
        gbm = GeometricBrownianMotion(S0=S0, r=r_dom, q=0.0, sigma=sigma_s)
        payoff_1d = VanillaPayoff(strike=K, option_type="put")
        engine_1d = LeastSquaresMonteCarlo(gbm, payoff_1d, LaguerrePolynomials(degree=3))
        price_1d, _ = engine_1d.pricer(T=T, n_steps=50, n_paths=2000, rng=rng_1)
        
        # 2. Quanto Process (Forced parameters to perfectly match 1D drift and mean-reversion)
        quanto = QuantoStochasticRatesProcess(
            S0=S0, rd0=r_dom, rf0=r_dom, q=0.0, 
            sigma_s=sigma_s, sigma_fx=0.1, rho_sfx=0.0,
            a_d=1.0, b_d=r_dom, sigma_d=0.00001, 
            a_f=1.0, b_f=r_dom, sigma_f=0.00001
        )
        
        payoff_3d = CompositePut(strike=K, column=0)
        features = QuantoRateFeatures(strike=K)
        engine_quanto = LeastSquaresMonteCarlo(quanto, payoff_3d, LaguerrePolynomials(degree=3))
        price_quanto, _ = engine_quanto.pricer(T=T, n_steps=50, n_paths=2000, rng=rng_2, create_features=features)
        
        # 3. Assertion (Using delta for MC variance differences)
        self.assertAlmostEqual(
            price_1d, price_quanto, delta=0.5, 
            msg="Quanto with 0 correlation must match standard GBM within MC noise!"
        )
    
    def test_quanto_american_parity(self):
        """Requirement: American Price must be >= European Price for Quanto."""
        process = QuantoGBM(S0=100, r_dom=0.05, r_for=0.02, q=0.0, sigma_s=0.2, sigma_fx=0.1, rho_sfx=0.3)
        payoff = CompositePut(strike=100, column=0)
        engine = LeastSquaresMonteCarlo(process, payoff, LaguerrePolynomials(3))
    
        p_amer, _ = engine.pricer(T=1.0, n_steps=50, n_paths=5000, rng=np.random.default_rng(42))
        p_euro, _ = engine.pricer(T=1.0, n_steps=50, n_paths=5000, exercise_times=[1.0], rng=np.random.default_rng(42))
    
        # American must be at least the value of European
        self.assertGreaterEqual(p_amer, p_euro - 0.05) 

    def test_stochastic_rates_consistency(self):
        """
        Consistency Check: If interest rate volatilities are zero, 
        QuantoStochasticRatesProcess must match QuantoGBM.
        """
        common_params = {
            "S0": 100, "rd0": 0.05, "rf0": 0.02, "q": 0.0, 
            "sigma_s": 0.2, "sigma_fx": 0.1, "rho_sfx": 0.3
        }
    
        stoch_proc = QuantoStochasticRatesProcess(
            **common_params, 
            a_d=1.0, b_d=0.05, sigma_d=0.0, 
            a_f=1.0, b_f=0.02, sigma_f=0.0  
        )
    
        gbm_proc = QuantoGBM(
            S0=100, r_dom=0.05, r_for=0.02, q=0.0, 
            sigma_s=0.2, sigma_fx=0.1, rho_sfx=0.3
        )
    
        payoff = CompositePut(strike=100, column=0)
        basis = LaguerrePolynomials(3)
    
        price_stoch, _ = LeastSquaresMonteCarlo(stoch_proc, payoff, basis).pricer(
            T=1.0, n_steps=20, n_paths=5000, rng=np.random.default_rng(42)
        )
    
        price_gbm, _ = LeastSquaresMonteCarlo(gbm_proc, payoff, basis).pricer(
            T=1.0, n_steps=20, n_paths=5000, rng=np.random.default_rng(42)
        )
    
        # Relaxed tolerance for 2D vs 3D noise generation
        self.assertTrue(abs(price_stoch - price_gbm) < 0.15, 
                        f"Stochastic rates logic {price_stoch} diverged from GBM {price_gbm}")

    def test_fixed_rate_quanto_pricer_runs(self):
        payoff = QuantoPut(strike=100, fx_fix=1.2)

        process = QuantoGBM(
            S0=100, r_dom=0.05, r_for=0.02, q=0.0,
            sigma_s=0.2, sigma_fx=0.1, rho_sfx=0.3,
        )

        engine = LeastSquaresMonteCarlo(
            process=process, payoff_function=payoff,
            basis_function=LaguerrePolynomials(degree=3),
        )

        price, stderr = engine.pricer(
            T=1.0, n_steps=10, n_paths=1000,
            rng=np.random.default_rng(42), use_antithetic=True,
        )

        self.assertTrue(np.isfinite(price))
        self.assertTrue(np.isfinite(stderr))
        self.assertGreaterEqual(price, 0.0)
        self.assertGreaterEqual(stderr, 0.0)

    def test_stochastic_rate_quanto_pricer_runs(self):
        payoff = QuantoPut(strike=100, fx_fix=1.2, column=0)

        process = QuantoStochasticRatesProcess(
            S0=100, rd0=0.05, rf0=0.02, q=0.0,
            sigma_s=0.2, sigma_fx=0.1, rho_sfx=0.3,
            a_d=1.0, b_d=0.05, sigma_d=0.01,
            a_f=1.0, b_f=0.02, sigma_f=0.01,
        )

        engine = LeastSquaresMonteCarlo(
            process=process, payoff_function=payoff,
            basis_function=LaguerrePolynomials(degree=3),
        )

        price, stderr = engine.pricer(
            T=1.0, n_steps=10, n_paths=1000,
            rng=np.random.default_rng(43), use_antithetic=True,
            create_features=QuantoRateFeatures(strike=100),
        )

        self.assertTrue(np.isfinite(price))
        self.assertTrue(np.isfinite(stderr))
        self.assertGreaterEqual(price, 0.0)
        self.assertGreaterEqual(stderr, 0.0)

    def test_stochastic_quanto_paths_and_discount_shape(self):
        process = QuantoStochasticRatesProcess(
            S0=100, rd0=0.05, rf0=0.02, q=0.0,
            sigma_s=0.2, sigma_fx=0.1, rho_sfx=0.3,
            a_d=1.0, b_d=0.05, sigma_d=0.01,
            a_f=1.0, b_f=0.02, sigma_f=0.01,
        )

        _, paths = process.simulate(
            T=1.0, n_steps=10, n_paths=100,
            rng=np.random.default_rng(44), use_antithetic=True,
        )

        discount = process.discount_step(paths, t=0, dt=0.1)

        self.assertEqual(paths.shape, (100, 11, 3))
        self.assertEqual(discount.shape, (100,))
        self.assertTrue(np.all(np.isfinite(discount)))
        self.assertTrue(np.all(discount > 0.0))

if __name__ == "__main__":
    unittest.main()