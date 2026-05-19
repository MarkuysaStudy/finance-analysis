# Machine Learning in Asset Management

---

## 1. Project files

| File               | Purpose |
|--------------------|---|
| `main.py`          | Main Python script that reproduces the empirical analysis. |
| `requirements.txt` | Python dependencies required to run the script. |
| `Makefile`         | Convenience commands for installation, execution, cleanup, and archiving. |
| `README.md`        | This instruction file. |

The script creates an output folder with two subfolders:

```text
outputs/
├── tables/
└── figures/
```

---

## 2. Empirical setting

The analysis uses monthly adjusted-close data for U.S. sector ETFs and the SPY benchmark.

### Sector ETFs

| Ticker | Sector |
|---|---|
| XLB | Materials |
| XLE | Energy |
| XLF | Financials |
| XLI | Industrials |
| XLK | Technology |
| XLP | Consumer Staples |
| XLU | Utilities |
| XLV | Health Care |
| XLY | Consumer Discretionary |

### Benchmark ETF

| Ticker | Role |
|---|---|
| SPY | S&P 500 benchmark |

Default sample period:

```text
2015-01-01 to 2025-12-31
```

Default out-of-sample evaluation starts after a 5-year initial training window.

---

## 3. What the script does

The workflow performs the following steps:

1. Downloads adjusted-close ETF prices from Yahoo Finance through `yfinance`.
2. Resamples prices to month-end frequency.
3. Computes monthly returns.
4. Builds lagged and rolling features:
   - 1-month, 3-month, 6-month, and 12-month sector returns;
   - 3-month, 6-month, and 12-month rolling volatility;
   - market return and market volatility features based on SPY;
   - 3-month relative strength versus SPY;
   - 12-month beta and correlation versus SPY;
   - 6-month drawdown proxy.
5. Runs expanding-window out-of-sample forecasts using:
   - Historical Mean benchmark;
   - Ridge Regression;
   - Random Forest;
   - Gradient Boosting.
6. Converts forecasts into monthly Top-K sector portfolios.
7. Computes gross and net portfolio performance metrics.
8. Exports forecast metrics, portfolio metrics, turnover, selection frequencies, robustness outputs, and figures.

---

## 4. Installation

### Option A: using Makefile

```bash
make install
```

This creates a local virtual environment in `.venv/` and installs all dependencies from `requirements.txt`.

### Option B: manual installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. How to run the analysis

### Default run

```bash
make run
```

The default configuration uses:

```text
start date: 2015-01-01
end date: 2025-12-31
initial training window: 5 years
Top-K selection rule: Top-3
annual risk-free rate: 2%
```

---

## 6. Useful Makefile commands

| Command | Description |
|---|---|
| `make install` | Create virtual environment and install dependencies. |
| `make run` | Run the empirical workflow with default parameters. |
| `make run-top2` | Run the workflow using Top-2 sector selection. |
| `make run-top4` | Run the workflow using Top-4 sector selection. |
| `make run-invvol` | Run the workflow using inverse-volatility weighting, if supported by the script. |
| `make clean` | Remove generated outputs and cache files. |
| `make zip` | Create a ZIP archive with the script, README, Makefile, requirements, and outputs. |
| `make help` | Show available Makefile commands. |

---

## 7. Main script arguments

The script supports command-line arguments.

```bash
python main.py \
  --start 2015-01-01 \
  --end 2025-12-31 \
  --train-years 5 \
  --top-k 3 \
  --rf 0.02 \
  --tcost-bps 10 \
  --weighting equal \
  --outdir outputs
```

### Arguments

| Argument | Default | Description |
|---|---:|---|
| `--start` | `2015-01-01` | Start date for downloading market data. |
| `--end` | `2025-12-31` | End date for downloading market data. |
| `--train-years` | `5` | Initial expanding-window training period in years. |
| `--top-k` | `3` | Number of sectors selected each month. |
| `--rf` | `0.02` | Annual risk-free rate used for Sharpe and Sortino ratios. |
| `--tcost-bps` | depends on script version | Transaction cost in basis points multiplied by monthly turnover. |
| `--weighting` | `equal` | Weighting method inside selected sectors. Supported values may include `equal` and `invvol`. |
| `--outdir` | `outputs` | Directory where tables and figures are saved. |

---

## 8. Output tables

The script saves CSV tables in:

```text
outputs/tables/
```

Typical outputs include:

| Output file | Description |
|---|---|
| `monthly_returns.csv` | Monthly return dataset for all ETFs. |
| `model_forecasts.csv` | Out-of-sample forecasts by date, ticker, and model. |
| `forecast_metrics.csv` | RMSE, MAE, directional accuracy, and forecast correlation. |
| `forecast_metrics_dm.csv` | Diebold–Mariano tests versus the Historical Mean benchmark. |
| `portfolio_returns_gross.csv` | Monthly gross portfolio returns. |
| `portfolio_returns_net.csv` | Monthly net portfolio returns after turnover-cost adjustment. |
| `portfolio_performance_gross.csv` | Gross CAGR, annualized return, volatility, Sharpe, drawdown, etc. |
| `portfolio_performance_net.csv` | Net performance metrics. |
| `turnover_summary.csv` | Average monthly turnover by active strategy. |
| `selection_frequency.csv` | Sector selection counts across model-based Top-K portfolios. |
| `year_end_selections.csv` | Top-K sector selections at December rebalancing dates. |
| `calendar_year_returns.csv` | Calendar-year realized returns by strategy. |
| `strategy_return_distribution.csv` | Distributional statistics of monthly strategy returns. |
| `descriptive_statistics_monthly_returns.csv` | Descriptive statistics for monthly ETF returns. |
| `descriptive_statistics_targets.csv` | Descriptive statistics for the model target return series. |
| `descriptive_statistics_features.csv` | Descriptive statistics for engineered features. |
| `feature_importance_raw.csv` | Raw feature-importance snapshots for tree-based models, if available. |

---

## 9. Output figures

The script saves PNG figures in:

```text
outputs/figures/
```

Typical figures include:

| Output file | Description |
|---|---|
| `01_correlation_heatmap.png` | ETF monthly return correlation heatmap. |
| `02_cumulative_wealth_gross.png` | Gross cumulative wealth paths of strategies. |
| `03_forecast_metric_bars.png` | Forecast metric comparison by model. |
| `04_strategy_performance_heatmap.png` | Strategy performance heatmap. |
| `05_feature_importance.png` | Feature-importance plot for tree-based models. |
| `06_sector_selection_frequency.png` | Sector selection frequency chart. |

---

## 10. Reproducing the term paper tables

To reproduce the tables used in the term paper, run:

```bash
make clean
make install
make run
```

Then use the CSV files from:

```text
outputs/tables/
```

The most important files for the main empirical section are:

```text
forecast_metrics.csv
portfolio_performance_gross.csv
turnover_summary.csv
portfolio_performance_net.csv
selection_frequency.csv
calendar_year_returns.csv
```

---

## 11. Notes on reproducibility

The script uses `yfinance`, which downloads data from Yahoo Finance. Therefore:

- an internet connection is required;
- the exact results may slightly change if Yahoo Finance revises historical adjusted prices;
- for exact archival reproducibility, save the generated `monthly_returns.csv` together with the paper.

The machine-learning models use fixed random seeds where applicable, so model outputs should be stable when the same input data and package versions are used.

---

## 12. Troubleshooting

### `ModuleNotFoundError`

Install dependencies:

```bash
pip install -r requirements.txt
```

or run:

```bash
make install
```

### No data downloaded

Check internet access and retry:

```bash
python main.py --outdir outputs
```

### Permission denied when running Makefile commands

Use:

```bash
make run
```

instead of executing the Makefile directly.

### Virtual environment is not activated

Activate it manually:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

---



## 13. Recommended execution order

```bash
make install
make clean
make run
make zip
```

After that, the generated outputs and archive can be submitted together with the term paper.
