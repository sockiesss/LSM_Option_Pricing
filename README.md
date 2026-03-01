# Least Squares Monte Carlo Option Pricing

## 1. Git Collaboration Workflow

To ensure code quality and project stability, please follow this workflow:

* **Branching**: Never push directly to the `main` branch. Create a new branch for every feature or fix (e.g., `feat-gbm-optimization` or `fix-regression-bug`).
* **Pull & Sync**: Before starting your work, run `git pull` to ensure your local environment is up to date with the remote repository.
* **Commit**: Stage your changes using `git add` and commit with a clear message following our format.
  * **Commit Message Format**：
    * **feat**: Adds a new feature or core functionality.
    * **fix**: Fixes a bug or an error in the code.
    * **docs**: Updates the documentation, such as this README.
    * **test**: Adds or updates testing scripts.
* **Push & Pull Request (PR)**: Push your branch to GitHub and open a Pull Request.
* **Code Review**: At least one other team member must review and approve your PR before it can be merged into `main`.

## 2. Tentative Project Plan
### Phase 1.1 Benchmarking & Testing (1 Member)
- Build benchmarks using LSM and other models like the Binomial Tree with other libraries. 
- Run sanity checks and replicate the numerical examples from the Longstaff and Schwartz (2001) paper. 
- Compare our engine's accuracy, speed, and memory usage against established benchmarks.

*Note: Benchmarking & Testing is continuous throughout the project.*

### Phase 1.2 Core Engine (2 Members)
Build the foundational Longstaff-Schwartz Least Squares Monte Carlo (LSM) algorithm. This includes writing the backward induction loop, the geometric Brownian motion simulator, and the American Put payoff structure.

### Phase 2: Performance Optimization (2 Members)
Profile the existing code and implement speedups. This involves writing variance reduction techniques (e.g., antithetic variates) and testing faster matrix solvers (e.g., Cholesky decomposition) for the regression step.

### Phase 3: Model Extensions (Whole Team, on a volunteer basis)
Expand the pricing engine to support alternative stochastic processes (like the Ornstein-Uhlenbeck process for commodities) and more complex exotic payoffs (such as Rainbow options or Asian options).
