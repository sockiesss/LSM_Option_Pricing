# American Option Pricing Via Least Squares Monte Carlo

## Project Goal
This project implements a high-performance pricing engine for American and Bermudan options using the **Least Squares Monte Carlo (LSM)** approach. 

The primary focus is on enhancing the standard Longstaff-Schwartz (2001) method with advanced variance reduction techniques, specifically **Control Variates** and **Leave-One-Out (LOO)** regression to reduce pricing error and in-sample bias. The engine can also be applied to other payoff structures, such as multi-asset max American calls, swing options, quantos, etc.

## Features
* **Core LSM Algorithm**: Backward induction with regression-based continuation value estimation.
* **Variance and Bias Reduction**: Antithetic variates, control variates (European options sampled at maturity or exercise times), and Leave-One-Out (LOO) cross-validation to eliminate look-ahead bias.
* **Regression Bases**: Laguerre polynomials and power polynomials for basis functions.
* **Multi-Asset Support**: Handles correlated assets via Cholesky decomposition.
* **Flexible Payoffs**: Vanilla puts/calls, max calls, swing options.
* **Benchmarks**: Comparison against Binomial Trees, Finite Difference Methods (QuantLib), and Black-Scholes.
* **Performance**: Optimized for speed and accuracy with configurable paths and steps.


## Project Structure
```text
├── .github/
│   └── workflows/
│       ├── ci.yml              # GitHub Actions automated testing
│       └── publish.yml         # PyPI publishing pipeline
├── LSM/
│   ├── __init__.py
│   ├── algorithms.py       # Core LeastSquaresMonteCarlo class
│   ├── binomial_tree.py    # CRR Binomial Tree for benchmarking
│   ├── control_variate.py  # Black-Scholes European prices and control variate logic
│   ├── payoffs.py          # Payoff classes (Vanilla, MaxCall, Swing)
│   ├── regression_bases.py # Laguerre and Power polynomial bases
│   └── stochastic_processes.py # GBM simulation with correlations
├── notebooks/
│   ├── demo.ipynb          # Quick start and Colab demonstration
│   └── tests.ipynb         # Benchmark tests and advanced payoffs
├── tests/
│   └── test_lsm.py         # Pytest suite for CI/CD
├── pyproject.toml          # Build and dependency configuration
└── README.md

```


## Installation

You can install the package via pip:

```bash
pip install lsm-option-pricing

```

## Quick Start

Import the modules and create an LSM engine:

```python
import numpy as np
from LSM.stochastic_processes import GeometricBrownianMotion
from LSM.payoffs import VanillaPayoff
from LSM.regression_bases import LaguerrePolynomials
from LSM.algorithms import LeastSquaresMonteCarlo

# Set up process, payoff, and basis
gbm = GeometricBrownianMotion(S0=36.0, r=0.06, q=0.0, sigma=0.2)
payoff = VanillaPayoff(strike=40.0, option_type="put")
basis = LaguerrePolynomials(degree=3)

# Create LSM engine and price option
lsm = LeastSquaresMonteCarlo(process=gbm, payoff_function=payoff, basis_function=basis)
price, stderr = lsm.pricer(T=1.0, n_steps=50, n_paths=10000)

print(f"American Put Price: {price:.4f} ± {stderr:.4f}")

```

## API Reference

### `LeastSquaresMonteCarlo.pricer()`

Prices the option using the standard Least Squares Monte Carlo algorithm. Evaluates the option by comparing immediate intrinsic value against the conditional expected continuation value.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `T` | `float` | Required | Time to maturity in years. |
| `n_steps` | `int` | Required | Number of discrete time steps for the simulation. |
| `n_paths` | `int` | Required | Number of Monte Carlo paths to generate (generates `n_paths/2` pairs if `use_antithetic=True`). |
| `rng` | `np.random.Generator` | `None` | NumPy random number generator instance for reproducible paths. |
| `use_antithetic` | `bool` | `False` | If `True`, uses antithetic variates for variance reduction. |
| `control_variate` | `str` | `None` | European option control variate method. Options: `'european_at_maturity'`, `'european_at_exercise'`, or `None`. |
| `create_features` | `Callable` | `None` | Function to create custom basis features for regression (e.g., cross-terms for multi-asset or Quanto options). |
| `cache` | `bool` | `False` | If `True`, caches the cash flow matrix allowing retrieval via `get_cashflow()`. |
| `exercise_times` | `array-like` | `None` | Specific exercise times for Bermudan options (e.g., `[0.25, 0.5, 1.0]`). If `None`, assumes an American option (exercisable at every step). |
| `simulation_times` | `array-like` | `None` | Custom time grid passed directly to the simulator. If provided, overrides `T` and `n_steps`. |
| `use_loo` | `bool` | `False` | If `True`, applies Leave-One-Out (LOO) cross-validation to reduce in-sample regression bias. |


### `LeastSquaresMonteCarlo.swing_pricer()`

Prices a natural gas or electricity swing option with specific volume constraints. Assumes every step in the simulation grid is a valid daily exercise opportunity.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `T` | `float` | Required | Total time to maturity in years. |
| `n_steps` | `int` | Required | Number of discrete time steps. |
| `n_paths` | `int` | Required | Number of Monte Carlo paths to simulate. |
| `rng` | `np.random.Generator` | `None` | NumPy random number generator instance for reproducible paths. |
| `use_antithetic` | `bool` | `False` | If `True`, uses antithetic variates for variance reduction. |
| `contract_prices` | `np.ndarray` | `None` | 1D array of shape `(n_steps + 1,)` representing the fixed strike price or forward curve value at each time step. |
| `simulation_times` | `np.ndarray` | `None` | Custom time grid. Overrides `T` and `n_steps`. Must exactly match the length of `contract_prices`. |
| `DCQ` | `float` | `1.0` | Daily Contract Quantity (the maximum volume allowed per single exercise). |
| `Ed` | `int` | `1` | Total number of exercise rights available (Annual Contract Quantity / DCQ). |
| `ToP_rights` | `int` | `0` | Minimum number of times the option MUST be exercised to avoid Take-or-Pay penalties. |

> **Note:**
> * **"Bang-Bang" Exercise:** Decisions are strictly all-or-nothing (0 or exactly `DCQ`). Partial volume exercises are not supported.
> * **Hard ToP Penalties:** Failing to meet `ToP_rights` invalidates the simulation path (assigns `-inf` value) rather than applying a proportional cash penalty.
> * **No Operational Friction:** Assumes immediate exercise rights without advance notice periods, resting times, or dynamic capacity limits.

**Returns:** A tuple `(price, std_err)` containing the estimated option price and the standard error.




## Demo
An interactive demo showing error convergence and basic pricing is available here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sockiesss/LSM_Option_Pricing/blob/main/notebooks/demo.ipynb)

## Dependencies

* Python
* NumPy
* SciPy
* Pandas (for data handling)
* Matplotlib (for plotting)
* Jupyter (for notebooks)
* QuantLib (optional, for FDM benchmarks)


## License
This project is licensed under the MIT License (see the LICENSE file for details).

## References
- [Longstaff, F. A., and E. S. Schwartz (2001). "Valuing American Options by Simulation: A Simple Least-Squares Approach."](https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf): Seminal LSM paper.
- [Broadie, M., and P. Glasserman (2004). "A Stochastic Mesh Method for Pricing High-Dimensional American Options." Journal of Computational Finance.](https://www.columbia.edu/~mnb2/broadie/Assets/mesh_working_paper.pdf): Multi-asset American max call benchmarks.
- [Jaillet, P., Ronn, E. I., and S. Tompaidis (2004). "Valuation of Commodity-Based Swing Options."](https://pubsonline.informs.org/doi/10.1287/mnsc.1040.0240): American equivalency and European strip bounds for swing options.
- [Hanfeld, M., and S. Schlüter (2016). "Operating a swing option on today's gas markets: How least squares Monte Carlo works and why it is beneficial." Working Paper.](https://www.econstor.eu/bitstream/10419/146758/1/868308544.pdf): Detailed backward induction logic for the $q_{n,t}$ offtake levels and the 2D state grid used in our swing_pricer.
- [Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering.](https://www.springer.com/gp/book/9780387004518): Quantos and other payoff structures, critique about LSM, etc.
- [Rasmussen, H. O. (2005). "Control Variates for American Options."](https://ideas.repec.org/a/rsk/journ0/2160484.html): Control variates for American options sampled at exercise.
- [Woo, R., et al. (2019). "Leave-one-out Least Squares Monte Carlo."](https://arxiv.org/abs/1810.02071): Leave-one-out LSM.
- [GitHub: luphord/longstaff_schwartz. "An implementation of the Longstaff-Schwartz algorithm for American option pricing."](https://github.com/luphord/longstaff_schwartz): Another LSM implementation.

## Acknowledgements
- Inspiration and repository structure: `luphord/longstaff_schwartz` (see References).
