1. Git Commit Format Example for README

feat: Adds a new feature or core functionality.
fix: Fixes a bug or an error in the code.
docs: Updates the documentation, such as the README.
test: Adds or updates testing scripts.

2. Tentative Project Plan
Phase 1 - Benchmarking & Testing (1 Member)
Build benchmarks using LSM and other models like the Binomial Tree with other libraries. Run sanity checks and replicate the numerical examples from the Longstaff and Schwartz (2001) paper. Compare our engine's accuracy, speed, and memory usage against established benchmarks.

Phase 1 - Core Engine (2 Members)
Build the foundational Longstaff-Schwartz Least Squares Monte Carlo (LSM) algorithm. This includes writing the backward induction loop, the geometric Brownian motion simulator, and the American Put payoff structure.

Phase 2: Performance Optimization (2 Members)
Profile the existing code and implement speedups. This involves writing variance reduction techniques (e.g., antithetic variates) and testing faster matrix solvers (e.g., Cholesky decomposition) for the regression step.

Phase 3: Model Extensions (Whole Team)
Expand the pricing engine to support alternative stochastic processes (like the Ornstein-Uhlenbeck process for commodities) and more complex exotic payoffs (such as Rainbow options or Asian options).

Note: Benchmarking & Testing is continuous throughout the project.