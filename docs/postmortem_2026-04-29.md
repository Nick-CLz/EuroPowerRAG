# Postmortem — Walk-forward 2025-10-31 → 2026-04-29

Generated: 2026-04-29T15:07:23.866310+00:00


## Results

- **DE_LU**: FAIL
  Trade log: `data/backtest/trades_01a36fcc.csv`

- **FR**: FAIL
  Trade log: `data/backtest/trades_1f99f51b.csv`

- **NL**: FAIL
  Trade log: `data/backtest/trades_44190834.csv`

- **GB**: FAIL
  Trade log: `data/backtest/trades_cc53c660.csv`


## Hypotheses to test next

- [ ] Tune XGBoost hyperparameters (grid search over max_depth, n_estimators)
- [ ] Expand ARIMA order search to (p,d,q) ∈ {0..3}³
- [ ] Wire real sentiment data (requires T0 human labelling)
- [ ] Investigate per-country volatility regimes
