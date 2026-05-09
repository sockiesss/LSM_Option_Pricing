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

Prices the option using the Least Squares Monte Carlo algorithm.

| Parameter | Type | Description |
| --- | --- | --- |
| `T` | float | Time to maturity in years. |
| `n_steps` | int | Number of time steps for the simulation. |
| `n_paths` | int | Number of Monte Carlo paths to generate. |
| `control_variate` | str or None | Variance reduction method (e.g., `'european_at_exercise'`). |
| `use_loo` | bool | Whether to apply Leave-One-Out bias reduction (default: `False`). |

**Returns:** A tuple `(price, std_err)` containing the estimated option price and the standard error.


## Demo
An interactive demo showing error convergence and basic pricing is available here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sockiesss/LSM_Option_Pricing/blob/main/notebooks/demo.ipynb)

## Dependencies

* Python 3.8+
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
