# American Option Pricing Via Least Squares Monte Carlo

## Project Goal
This project implements a high-performance pricing engine for American and Bermudan options using the **Least Squares Monte Carlo (LSM)** approach. The primary focus is on enhancing the standard Longstaff-Schwartz (2001) method with advanced variance reduction techniques, specifically **Control Variates** and **Leave-One-Out (LOO)** regression to reduce in-sample bias and pricing error.

The engine can also be applied to other payoff structures, such as multi-asset max American calls, swing options, and quantos.

## Features
- **Core LSM Algorithm**: Backward induction with regression-based continuation value estimation.
- **Variance Reduction**: Antithetic variates, control variates (European options sampled at maturity or exercise times).
- **Regression Bases**: Laguerre polynomials and power polynomials for basis functions.
- **Multi-Asset Support**: Handles correlated assets via Cholesky decomposition.
- **Flexible Payoffs**: Vanilla puts/calls, max calls, swing options.
- **Benchmarks**: Comparison against Binomial Trees, Finite Difference Methods (QuantLib), and Black-Scholes.
- **Performance**: Optimized for speed and accuracy with configurable paths and steps.

## Project Structure
```
├── LSM/
│   ├── __init__.py
│   ├── algorithms.py       # Core LeastSquaresMonteCarlo class
│   ├── binomial_tree.py    # CRR Binomial Tree for benchmarking
│   ├── control_variate.py  # Black-Scholes European prices and control variate logic
│   ├── payoffs.py          # Payoff classes (Vanilla, MaxCall, Swing)
│   ├── regression_bases.py # Laguerre and Power polynomial bases
│   └── stochastic_processes.py # GBM simulation with correlations
├── notebooks/
│   ├── tests.ipynb         # Benchmark tests vs Finite Difference Methods
│   └── analysis.ipynb      # Convergence and error analysis
├── data/                   # Benchmark result CSVs
└── README.md
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sockiesss/LSM_Option_Pricing.git
   cd LSM_Option_Pricing
   ```

2. Install dependencies:
   ```bash
   pip install numpy scipy pandas matplotlib jupyter
   ```
   Optional: For QuantLib benchmarks, install QuantLib-Python:
   ```bash
   pip install QuantLib
   ```

3. (Optional) Set up a virtual environment:
   ```bash
   python -m venv lsm_env
   source lsm_env/bin/activate  # On Windows: lsm_env\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage
Import the modules and create an LSM engine:

```python
from LSM.stochastic_processes import GeometricBrownianMotion
from LSM.payoffs import VanillaPayoff
from LSM.regression_bases import LaguerrePolynomials
from LSM.algorithms import LeastSquaresMonteCarlo

# Define parameters
S0 = 36.0
K = 40.0
r = 0.06
q = 0.0
sigma = 0.2
T = 1.0
n_steps = 50
n_paths = 10000

# Set up process, payoff, and basis
gbm = GeometricBrownianMotion(S0=S0, r=r, q=q, sigma=sigma)
payoff = VanillaPayoff(strike=K, option_type="put")
basis = LaguerrePolynomials(degree=3)

# Create LSM engine
lsm = LeastSquaresMonteCarlo(process=gbm, payoff_function=payoff, basis_function=basis)

# Price the option
price, stderr = lsm.pricer(T=T, n_steps=n_steps, n_paths=n_paths)
print(f"American Put Price: {price:.4f} ± {stderr:.4f}")
```

For control variates:
```python
price_cv, stderr_cv = lsm.pricer(T=T, n_steps=n_steps, n_paths=n_paths, control_variate='european_at_exercise')
```

For swing options:
```python
import numpy as np
from LSM.payoffs import SwingSpread
from LSM.regression_bases import PowerPolynomials

# Define swing payoff and contract prices
swing_payoff = SwingSpread(option_type="call")  # Spread = S_t - contract_price
contract_prices = np.linspace(50, 55, n_steps + 1)  # Example forward curve

# Use power-polynomial basis (safe for negative spreads)
basis_swing = PowerPolynomials(degree=3)

# Create LSM engine for swing
lsm_swing = LeastSquaresMonteCarlo(process=gbm, payoff_function=swing_payoff, basis_function=basis_swing)

# Price swing option
price_swing, stderr_swing = lsm_swing.swing_pricer(
   T=T, n_steps=n_steps, n_paths=n_paths, 
   contract_prices=contract_prices, DCQ=1.0, Ed=5, ToP_rights=2
)
print(f"Swing Option Price: {price_swing:.4f} ± {stderr_swing:.4f}")
```

## Examples
- **Basic American Put**: See `notebooks/tests.ipynb` for sanity checks and benchmarks.
- **Table 1 Replication**: Replicates Longstaff-Schwartz (2001) Table 1 with FDM comparisons.
- **Random Benchmarks**: Tests across various parameters for accuracy.
- **Moneyness Analysis**: Evaluates performance for ITM/ATM/OTM options.

Run the notebooks:
```bash
jupyter notebook notebooks/tests.ipynb
```

## Dependencies
- Python 3.8+
- NumPy
- SciPy
- Pandas (for data handling)
- Matplotlib (for plotting)
- Jupyter (for notebooks)
- QuantLib (optional, for FDM benchmarks)

## Testing
Run benchmarks in `notebooks/tests.ipynb` to verify against known results.

## Contributing
Contributions are welcome! Please fork the repo, make changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## References
- [Longstaff-Schwartz (2001)](https://academic.oup.com/rfs/article-abstract/14/1/113/1606182): Original LSM paper.
- [Rasmussen (2005)](https://www.sciencedirect.com/science/article/pii/S0378426604001514): Control variates for American options.
- [Woo et al. (2019)](https://www.sciencedirect.com/science/article/pii/S0167947318301788): Leave-one-out LSM.
- [Glasserman (2004)](https://www.springer.com/gp/book/9780387004518): Monte Carlo Methods in Financial Engineering.
- [GitHub: luphord/longstaff_schwartz](https://github.com/luphord/longstaff_schwartz): Another LSM implementation.

## Acknowledgements
- Inspiration and repository structure: `luphord/longstaff_schwartz` (see References).
