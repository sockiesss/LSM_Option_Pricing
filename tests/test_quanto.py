import numpy as np

from LSM.algorithms import LeastSquaresMonteCarlo
from LSM.payoffs import VanillaPayoff, FixedFXQuantoPayoff, StateColumnPayoff, QuantoRateFeatures
from LSM.regression_bases import LaguerrePolynomials
from LSM.stochastic_processes import QuantoGBM, QuantoStochasticRatesProcess


def test_fixed_rate_quanto_pricer_runs():
    base_put = VanillaPayoff(strike=100, option_type="put")
    payoff = FixedFXQuantoPayoff(base_put, fx_fix=1.2)

    process = QuantoGBM(
        S0=100,
        r_dom=0.05,
        r_for=0.02,
        q=0.0,
        sigma_s=0.2,
        sigma_fx=0.1,
        rho_sfx=0.3,
    )

    engine = LeastSquaresMonteCarlo(
        process=process,
        payoff_function=payoff,
        basis_function=LaguerrePolynomials(degree=3),
    )

    price, stderr = engine.quanto_pricer(
        T=1.0,
        n_steps=10,
        n_paths=1000,
        rng=np.random.default_rng(42),
        use_antithetic=True,
    )

    assert np.isfinite(price)
    assert np.isfinite(stderr)
    assert price >= 0.0
    assert stderr >= 0.0


def test_stochastic_rate_quanto_pricer_runs():
    base_put = VanillaPayoff(strike=100, option_type="put")
    payoff = StateColumnPayoff(base_put, column=0)

    process = QuantoStochasticRatesProcess(
        S0=100,
        rd0=0.05,
        rf0=0.02,
        q=0.0,
        sigma_s=0.2,
        sigma_fx=0.1,
        rho_sfx=0.3,
        a_d=1.0,
        b_d=0.05,
        sigma_d=0.01,
        a_f=1.0,
        b_f=0.02,
        sigma_f=0.01,
    )

    engine = LeastSquaresMonteCarlo(
        process=process,
        payoff_function=payoff,
        basis_function=LaguerrePolynomials(degree=3),
    )

    price, stderr = engine.quanto_pricer(
        T=1.0,
        n_steps=10,
        n_paths=1000,
        rng=np.random.default_rng(43),
        use_antithetic=True,
        create_features=QuantoRateFeatures(strike=100),
    )

    assert np.isfinite(price)
    assert np.isfinite(stderr)
    assert price >= 0.0
    assert stderr >= 0.0


def test_stochastic_quanto_paths_and_discount_shape():
    process = QuantoStochasticRatesProcess(
        S0=100,
        rd0=0.05,
        rf0=0.02,
        q=0.0,
        sigma_s=0.2,
        sigma_fx=0.1,
        rho_sfx=0.3,
        a_d=1.0,
        b_d=0.05,
        sigma_d=0.01,
        a_f=1.0,
        b_f=0.02,
        sigma_f=0.01,
    )

    _, paths = process.simulate(
        T=1.0,
        n_steps=10,
        n_paths=100,
        rng=np.random.default_rng(44),
        use_antithetic=True,
    )

    discount = process.discount_step(paths, t=0, dt=0.1)

    assert paths.shape == (100, 11, 3)
    assert discount.shape == (100,)
    assert np.all(np.isfinite(discount))
    assert np.all(discount > 0.0)
