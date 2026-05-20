
"""

Tables (CSV, saved to <outdir>/tables/)
- monthly_returns.csv                          : monthly returns for all ETFs
- forecast_metrics.csv                         : RMSE/MAE/directional accuracy/correlation (Table 4)
- forecast_metrics_dm.csv                      : Diebold–Mariano tests vs HistMean (optional; Appendix)
- portfolio_returns_gross.csv                  : monthly strategy returns (gross)
- portfolio_returns_net.csv                    : monthly strategy returns (net of turnover costs)
- portfolio_performance_gross.csv              : CAGR/Sharpe/Drawdown/... (Table 5 / Appendix D1)
- portfolio_performance_net.csv                : net-of-cost performance for the selected turnover-cost assumption
  (baseline in the paper: 10 bps; stress scenario: rerun with --tcost-bps 50)
- turnover_summary.csv                         : average monthly turnover (Table 6)
- strategy_return_distribution.csv             : distributional statistics (Appendix E)
- selection_frequency.csv                      : sector selection counts by model (Table 8)
- year_end_selections.csv                      : Top-K selections at December rebalancing dates (Appendix C1)
- calendar_year_returns.csv                    : calendar-year portfolio returns (Appendix C2)
- descriptive_statistics_monthly_returns.csv    : descriptive stats for returns (Appendix F)
- descriptive_statistics_targets.csv            : descriptive stats for model targets (Appendix G)
- descriptive_statistics_features.csv           : descriptive stats for engineered features (Appendix)

Figures (PNG, saved to <outdir>/figures/)
- 01_correlation_heatmap.png
- 02_cumulative_wealth_gross.png
- 03_forecast_metric_bars.png
- 04_strategy_performance_heatmap.png
- 05_feature_importance.png
- 06_sector_selection_frequency.png

"""
from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Please install yfinance (see requirements file).") from e






DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_TRAIN_YEARS = 5
DEFAULT_TOP_K = 3
DEFAULT_RF_ANNUAL = 0.02
DEFAULT_TCOST_BPS = 10.0

TICKERS: Dict[str, str] = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "SPY": "S&P 500 (benchmark)",
}






def ensure_dirs(outdir: Path) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"
    tabdir = outdir / "tables"
    figdir.mkdir(exist_ok=True)
    tabdir.mkdir(exist_ok=True)
    return figdir, tabdir


def pct(x: float) -> float:
    """Convert fraction to percent."""
    return 100.0 * x






def download_monthly_prices(
    tickers: List[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download adjusted price series (auto_adjust=True) and resample to month-end.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})


    prices = prices.resample("M").last().dropna(how="all")
    prices = prices.dropna(axis=1, how="all")
    return prices


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Simple monthly returns computed from month-end prices.
    """
    return prices.pct_change().dropna()






def make_asset_features(returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build a compact, economically interpretable feature set for each asset i.
    All features are lagged (shifted) so that only information available at t-1 or earlier is used
    to forecast the return at time t.

    Features (used in the report):
    - ret1m, ret3m, ret6m, ret12m   : lagged cumulative returns (simple sums of monthly returns)
    - vol3m, vol6m, vol12m           : rolling volatility (std)
    - mkt_ret1m, mkt_ret3m            : market lagged returns (SPY)
    - mkt_vol6m                        : market rolling volatility (SPY)
    - rel_strength3m                   : ret_3m - mkt_ret_3m
    - beta12m, corr12m                : rolling beta and correlation vs SPY (12 months)
    - drawdown6m                       : minimum drawdown over the last 6 months

    Target:
    - target = r_{i,t} (current month's return)
    """
    if "SPY" not in returns.columns:
        raise ValueError("Returns must include SPY column for market features.")

    market = returns["SPY"]
    features: Dict[str, pd.DataFrame] = {}

    for ticker in returns.columns:
        r = returns[ticker]
        df = pd.DataFrame(index=returns.index)


        df["ret1m"] = r.shift(1)
        df["ret3m"] = r.rolling(3).sum().shift(1)
        df["ret6m"] = r.rolling(6).sum().shift(1)
        df["ret12m"] = r.rolling(12).sum().shift(1)


        df["vol3m"] = r.rolling(3).std().shift(1)
        df["vol6m"] = r.rolling(6).std().shift(1)
        df["vol12m"] = r.rolling(12).std().shift(1)


        df["mkt_ret1m"] = market.shift(1)
        df["mkt_ret3m"] = market.rolling(3).sum().shift(1)
        df["mkt_vol6m"] = market.rolling(6).std().shift(1)


        df["rel_strength3m"] = df["ret3m"] - df["mkt_ret3m"]


        df["beta12m"] = r.rolling(12).cov(market).shift(1) / market.rolling(12).var().shift(1)
        df["corr12m"] = r.rolling(12).corr(market).shift(1)


        wealth = (1.0 + r.fillna(0.0)).cumprod()
        peak = wealth.cummax()
        drawdown = wealth / peak - 1.0
        df["drawdown6m"] = drawdown.rolling(6).min().shift(1)


        df["target"] = r


        features[ticker] = df.dropna()

    return features






def build_models() -> Dict[str, object]:
    """
    Model set described in the report:
    - HistMean (benchmark)
    - Ridge (regularized linear)
    - RF (random forest)
    - GB (gradient boosting)
    """
    models: Dict[str, object] = {
        "HistMean": None,
        "Ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "RF": RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42,
        ),
        "GB": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.10,
            max_depth=3,
            random_state=42,
        ),
    }
    return models


@dataclass
class ForecastResult:
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    dm_tests: pd.DataFrame
    cw_tests: pd.DataFrame
    feature_importance: pd.DataFrame


def _norm_cdf(x: float) -> float:
    """Normal CDF without scipy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def diebold_mariano_test(loss_diff: np.ndarray) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy (h=1).
    loss_diff: series d_t = L1_t - L0_t, e.g., squared error differential.

    Returns: (DM statistic, two-sided p-value with normal approximation).
    """
    d = loss_diff.astype(float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan

    mean_d = d.mean()
    var_d = d.var(ddof=1)
    if var_d <= 0:
        return np.nan, np.nan

    dm = mean_d / math.sqrt(var_d / n)
    p = 2.0 * (1.0 - _norm_cdf(abs(dm)))
    return dm, p


def clark_west_test(actual: np.ndarray, benchmark_forecast: np.ndarray, alternative_forecast: np.ndarray) -> Tuple[float, float]:
    """
    Clark-West-style adjusted test for nested forecast comparison.

    The test is used here as a diagnostic comparing an alternative forecast with the
    Historical Mean benchmark. A positive statistic indicates that the alternative
    forecast improves adjusted MSPE relative to the benchmark. The p-value is a
    one-sided normal-approximation p-value.
    """
    y = np.asarray(actual, dtype=float)
    f0 = np.asarray(benchmark_forecast, dtype=float)
    f1 = np.asarray(alternative_forecast, dtype=float)
    mask = np.isfinite(y) & np.isfinite(f0) & np.isfinite(f1)
    y, f0, f1 = y[mask], f0[mask], f1[mask]
    n = len(y)
    if n < 10:
        return np.nan, np.nan

    e0 = y - f0
    e1 = y - f1
    adjusted = e0 ** 2 - (e1 ** 2 - (f0 - f1) ** 2)
    var_adj = adjusted.var(ddof=1)
    if var_adj <= 0:
        return np.nan, np.nan
    stat = adjusted.mean() / math.sqrt(var_adj / n)
    p_one_sided = 1.0 - _norm_cdf(stat)
    return stat, p_one_sided


def rolling_forecasts(
    features: Dict[str, pd.DataFrame],
    train_years: int,
    window: str = "expanding",
    rolling_months: int = 36,
) -> ForecastResult:
    """
    Expanding-window or rolling-window OOS forecasts.

    Training begins after train_years*12 observations inside each asset's feature panel.
    With window='rolling', each model is estimated using the latest rolling_months
    observations available before the forecast date.
    """
    models = build_models()
    train_months = train_years * 12
    if window not in {"expanding", "rolling"}:
        raise ValueError("window must be either 'expanding' or 'rolling'")
    if rolling_months < 12:
        raise ValueError("rolling_months must be at least 12")

    rows = []
    fi_rows = []

    for ticker, df in features.items():
        X = df.drop(columns=["target"])
        y = df["target"]

        start_idx = train_months
        for i in range(start_idx, len(df)):
            if window == "rolling":
                train_start = max(0, i - rolling_months)
                train_X = X.iloc[train_start:i]
                train_y = y.iloc[train_start:i]
            else:
                train_X = X.iloc[:i]
                train_y = y.iloc[:i]

            test_X = X.iloc[[i]]
            test_y = float(y.iloc[i])
            dt = df.index[i]

            hist_mean = float(train_y.mean())
            rows.append({"date": dt, "ticker": ticker, "model": "HistMean", "actual": test_y, "forecast": hist_mean})

            for name, model in models.items():
                if name == "HistMean":
                    continue
                mdl = clone(model)
                mdl.fit(train_X, train_y)
                pred = float(mdl.predict(test_X)[0])
                rows.append({"date": dt, "ticker": ticker, "model": name, "actual": test_y, "forecast": pred})

                if dt.month == 12 and hasattr(mdl, "feature_importances_"):
                    for col, val in zip(train_X.columns, getattr(mdl, "feature_importances_")):
                        fi_rows.append({
                            "date": dt,
                            "ticker": ticker,
                            "model": name,
                            "feature": col,
                            "importance": float(val),
                        })

    forecasts = pd.DataFrame(rows).sort_values(["date", "ticker", "model"]).reset_index(drop=True)
    feature_importance = pd.DataFrame(fi_rows)

    metric_rows = []
    for model, g in forecasts.groupby("model"):
        rmse = math.sqrt(mean_squared_error(g["actual"], g["forecast"]))
        mae = mean_absolute_error(g["actual"], g["forecast"])
        direction = float(np.mean(np.sign(g["actual"]) == np.sign(g["forecast"])))
        corr = float(g[["actual", "forecast"]].corr().iloc[0, 1])
        metric_rows.append({"model": model, "RMSE": rmse, "MAE": mae, "DirectionalAccuracy": direction, "ForecastCorrelation": corr})
    metrics = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)

    base = forecasts[forecasts["model"] == "HistMean"].copy()
    base["se_base"] = (base["actual"] - base["forecast"]) ** 2

    dm_rows = []
    cw_rows = []
    for model in [m for m in forecasts["model"].unique() if m != "HistMean"]:
        alt = forecasts[forecasts["model"] == model].copy()
        alt["se_alt"] = (alt["actual"] - alt["forecast"]) ** 2
        merged = base.merge(
            alt[["date", "ticker", "forecast", "se_alt"]].rename(columns={"forecast": "forecast_alt"}),
            on=["date", "ticker"],
            how="inner",
        )
        d = (merged["se_alt"] - merged["se_base"]).to_numpy()
        dm, p = diebold_mariano_test(d)
        dm_rows.append({"model": model, "DM_stat": dm, "p_value": p, "n_obs": int(len(d))})

        cw_stat, cw_p = clark_west_test(
            merged["actual"].to_numpy(),
            merged["forecast"].to_numpy(),
            merged["forecast_alt"].to_numpy(),
        )
        cw_rows.append({"model": model, "CW_stat": cw_stat, "p_value_one_sided": cw_p, "n_obs": int(len(merged))})

    dm_tests = pd.DataFrame(dm_rows).sort_values("p_value").reset_index(drop=True)
    cw_tests = pd.DataFrame(cw_rows).sort_values("p_value_one_sided").reset_index(drop=True)

    return ForecastResult(
        forecasts=forecasts,
        metrics=metrics,
        dm_tests=dm_tests,
        cw_tests=cw_tests,
        feature_importance=feature_importance,
    )






def _equal_weight_topk(chosen: List[str]) -> Dict[str, float]:
    if not chosen:
        return {}
    w = 1.0 / len(chosen)
    return {t: w for t in chosen}


def _inverse_vol_weights(chosen: List[str], vol_series: pd.Series) -> Dict[str, float]:
    """
    Inverse-volatility weights (optional robustness).
    vol_series must be a Series indexed by ticker at the rebalance date.
    """
    v = vol_series.reindex(chosen).replace(0, np.nan).dropna()
    if len(v) == 0:
        return _equal_weight_topk(chosen)
    inv = 1.0 / v
    inv = inv / inv.sum()
    return inv.to_dict()



def _risk_parity_weights(chosen: List[str], cov: pd.DataFrame) -> Dict[str, float]:
    """
    Approximate long-only equal-risk-contribution weights for the selected assets.
    Falls back to inverse-volatility weights if the covariance matrix is not usable.
    """
    if not chosen:
        return {}
    cov = cov.reindex(index=chosen, columns=chosen).replace([np.inf, -np.inf], np.nan)
    if cov.isna().any().any() or cov.shape[0] == 0:
        vol = pd.Series(np.sqrt(np.diag(cov.fillna(0.0))), index=chosen).replace(0, np.nan)
        return _inverse_vol_weights(chosen, vol)

    sigma = cov.to_numpy(dtype=float)
    n = len(chosen)

    sigma = sigma + np.eye(n) * 1e-10
    w = np.repeat(1.0 / n, n)
    target = 1.0 / n

    for _ in range(500):
        port_var = float(w @ sigma @ w)
        if port_var <= 0 or not np.isfinite(port_var):
            return _equal_weight_topk(chosen)
        mrc = sigma @ w
        rc = w * mrc / port_var
        rc = np.clip(rc, 1e-8, None)
        w = w * (target / rc) ** 0.5
        w = np.clip(w, 1e-8, None)
        w = w / w.sum()

    return {t: float(wi) for t, wi in zip(chosen, w)}


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Monthly turnover defined as 0.5 * sum(|w_t - w_{t-1}|).
    """
    w = weights.fillna(0.0)
    dw = (w - w.shift(1)).abs()
    turnover = 0.5 * dw.sum(axis=1)
    return turnover


def build_portfolios(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    top_k: int,
    weighting: str = "equal",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build model-based and benchmark portfolios and return:
    - portfolios: long format with monthly strategy returns
    - weights: wide format with strategy weights (for turnover)

    Supported weighting rules:
    - equal: equal weights inside the selected Top-K set;
    - invvol: inverse trailing-volatility weights;
    - riskparity: approximate equal-risk-contribution weights using trailing covariance.
    """
    if weighting not in {"equal", "invvol", "riskparity"}:
        raise ValueError("weighting must be one of: equal, invvol, riskparity")

    asset_universe = [c for c in returns.columns if c != "SPY"]
    test_dates = sorted(set(forecasts["date"]))

    eq_rets = returns.loc[test_dates, asset_universe].mean(axis=1)
    spy_rets = returns.loc[test_dates, "SPY"]
    mom_signal = returns[asset_universe].rolling(12).sum().shift(1)

    port_rows = []
    weight_rows = []

    for dt in test_dates:
        port_rows.append({"date": dt, "strategy": "EqualWeight", "return": float(eq_rets.loc[dt])})
        port_rows.append({"date": dt, "strategy": "SPY", "return": float(spy_rets.loc[dt])})

    for dt in test_dates:
        ranked = mom_signal.loc[dt].dropna().sort_values(ascending=False).head(top_k)
        chosen = ranked.index.tolist()
        w = _equal_weight_topk(chosen)
        realized = float((returns.loc[dt, list(w.keys())] * pd.Series(w)).sum()) if w else np.nan
        port_rows.append({"date": dt, "strategy": f"MomentumTop{top_k}", "return": realized})
        row = {"date": dt, "strategy": f"MomentumTop{top_k}"}
        row.update({t: w.get(t, 0.0) for t in asset_universe})
        weight_rows.append(row)

    for model in sorted(forecasts["model"].unique()):
        for dt in test_dates:
            g = forecasts[(forecasts["date"] == dt) & (forecasts["model"] == model) & (forecasts["ticker"].isin(asset_universe))]
            ranked = g.sort_values("forecast", ascending=False).head(top_k)
            chosen = ranked["ticker"].tolist()

            if weighting == "invvol":
                vol6 = returns[asset_universe].rolling(6).std().shift(1).loc[dt]
                w = _inverse_vol_weights(chosen, vol6)
            elif weighting == "riskparity":
                hist = returns.loc[:dt, asset_universe].iloc[:-1].tail(12)
                cov = hist.cov()
                w = _risk_parity_weights(chosen, cov)
            else:
                w = _equal_weight_topk(chosen)

            realized = float((returns.loc[dt, list(w.keys())] * pd.Series(w)).sum()) if w else np.nan
            port_rows.append({"date": dt, "strategy": model, "return": realized})

            row = {"date": dt, "strategy": model}
            row.update({t: w.get(t, 0.0) for t in asset_universe})
            weight_rows.append(row)

    portfolios = pd.DataFrame(port_rows).sort_values(["date", "strategy"]).reset_index(drop=True)
    weights = pd.DataFrame(weight_rows).sort_values(["date", "strategy"]).reset_index(drop=True)

    return portfolios, weights






def performance_table(portfolios: pd.DataFrame, rf_annual: float) -> pd.DataFrame:
    """
    Standard performance metrics for monthly strategy return series.
    """
    rows = []
    rf_monthly = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0

    for strategy, g in portfolios.groupby("strategy"):
        s = g.sort_values("date")["return"].astype(float)
        wealth = (1.0 + s).cumprod()
        total_months = len(s)

        cagr = float(wealth.iloc[-1] ** (12.0 / total_months) - 1.0)
        ann_mean = float(s.mean() * 12.0)
        ann_vol = float(s.std(ddof=1) * math.sqrt(12.0))

        downside = float(s[s < rf_monthly].std(ddof=1) * math.sqrt(12.0)) if (s < rf_monthly).any() else np.nan
        sharpe = float((ann_mean - rf_annual) / ann_vol) if ann_vol and ann_vol > 0 else np.nan
        sortino = float((ann_mean - rf_annual) / downside) if downside and not np.isnan(downside) and downside > 0 else np.nan

        max_dd = float((wealth / wealth.cummax() - 1.0).min())
        calmar = float(cagr / abs(max_dd)) if max_dd != 0 else np.nan

        rows.append({
            "Strategy": strategy,
            "CAGR": cagr,
            "AnnualMean": ann_mean,
            "AnnualVol": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDrawdown": max_dd,
            "Calmar": calmar,
            "FinalWealth": float(wealth.iloc[-1]),
        })

    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)


def strategy_return_distribution(portfolios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in portfolios.groupby("strategy"):
        s = g.sort_values("date")["return"].astype(float)
        rows.append({
            "Strategy": strategy,
            "Months": float(len(s)),
            "MeanMonthlyReturn": float(s.mean()),
            "StdDev": float(s.std(ddof=1)),
            "Min": float(s.min()),
            "Max": float(s.max()),
            "PositiveMonthShare": float((s > 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("MeanMonthlyReturn", ascending=False).reset_index(drop=True)


def calendar_year_returns(portfolios: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar-year realized returns computed by compounding monthly strategy returns within a year.
    """
    df = portfolios.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    out = []
    for (strategy, year), g in df.groupby(["strategy", "year"]):
        s = g.sort_values("date")["return"].astype(float)
        yr = float((1.0 + s).prod() - 1.0)
        out.append({"Year": int(year), "Strategy": strategy, "CalendarYearReturn": yr})
    wide = pd.DataFrame(out).pivot(index="Year", columns="Strategy", values="CalendarYearReturn").sort_index()
    wide.reset_index(inplace=True)
    return wide


def year_end_selections(forecasts: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    Top-K tickers selected in December (based on forecasts at December month-end).
    """
    asset_universe = [t for t in forecasts["ticker"].unique() if t != "SPY"]
    f = forecasts.copy()
    f["year"] = pd.to_datetime(f["date"]).dt.year
    f["month"] = pd.to_datetime(f["date"]).dt.month

    out = []
    for (model, year), g in f[(f["month"] == 12) & (f["ticker"].isin(asset_universe))].groupby(["model", "year"]):
        top = g.sort_values("forecast", ascending=False).head(top_k)["ticker"].tolist()
        out.append({"Year": int(year), "Model": model, f"Top{top_k}": ", ".join(top)})
    return pd.DataFrame(out).sort_values(["Year", "Model"]).reset_index(drop=True)






def descriptive_statistics(data: pd.DataFrame, rf_annual: float) -> pd.DataFrame:
    """
    Descriptive statistics for each numeric column.
    Sharpe ratio is annualized from monthly data.
    """
    numeric = data.select_dtypes(include=[np.number]).copy()
    rf_monthly = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0

    rows = []
    for col in numeric.columns:
        s = numeric[col].dropna().astype(float)
        if len(s) == 0:
            continue
        mean = float(s.mean())
        std = float(s.std(ddof=1))
        sharpe = float(((mean - rf_monthly) / std) * math.sqrt(12.0)) if std > 0 else np.nan

        rows.append({
            "Indicator": col,
            "Number of observations": float(s.count()),
            "Mean": mean,
            "Median": float(s.median()),
            "Skewness": float(s.skew()),
            "Kurtosis": float(s.kurt()),
            "Min": float(s.min()),
            "Max": float(s.max()),
            "Standard Deviation": std,
            "Sharpe Ratio": sharpe,
        })

    return pd.DataFrame(rows)



def _annualized_sharpe(monthly_returns: np.ndarray, rf_annual: float) -> float:
    s = np.asarray(monthly_returns, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 3:
        return np.nan
    rf_monthly = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0
    ann_mean = s.mean() * 12.0
    ann_vol = s.std(ddof=1) * math.sqrt(12.0)
    if ann_vol <= 0:
        return np.nan
    return float((ann_mean - rf_annual) / ann_vol)


def block_bootstrap_portfolio_tests(
    portfolios: pd.DataFrame,
    comparisons: List[Tuple[str, str]],
    rf_annual: float,
    n_boot: int = 2000,
    block_size: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Block bootstrap diagnostics for Sharpe-ratio and median-return differences.
    Returns two-sided empirical p-values for the null of zero difference.
    """
    rng = np.random.default_rng(seed)
    wide = portfolios.pivot(index="date", columns="strategy", values="return").sort_index()
    n = len(wide)
    rows = []
    if n == 0:
        return pd.DataFrame()

    for left, right in comparisons:
        if left not in wide.columns or right not in wide.columns:
            continue
        pair = wide[[left, right]].dropna()
        if len(pair) < block_size + 1:
            continue
        x = pair[left].to_numpy(dtype=float)
        y = pair[right].to_numpy(dtype=float)
        obs_sharpe_diff = _annualized_sharpe(x, rf_annual) - _annualized_sharpe(y, rf_annual)
        obs_median_diff = float(np.median(x) - np.median(y))

        boot_sharpe = []
        boot_median = []
        m = len(pair)
        starts = np.arange(0, m)
        for _ in range(n_boot):
            idx = []
            while len(idx) < m:
                st = int(rng.choice(starts))
                idx.extend([(st + j) % m for j in range(block_size)])
            idx = np.array(idx[:m])
            xb = x[idx]
            yb = y[idx]
            boot_sharpe.append(_annualized_sharpe(xb, rf_annual) - _annualized_sharpe(yb, rf_annual))
            boot_median.append(float(np.median(xb) - np.median(yb)))

        boot_sharpe = np.asarray(boot_sharpe, dtype=float)
        boot_median = np.asarray(boot_median, dtype=float)

        sharpe_centered = boot_sharpe - np.nanmean(boot_sharpe)
        median_centered = boot_median - np.nanmean(boot_median)
        p_sharpe = float(np.nanmean(np.abs(sharpe_centered) >= abs(obs_sharpe_diff)))
        p_median = float(np.nanmean(np.abs(median_centered) >= abs(obs_median_diff)))

        rows.append({
            "comparison": f"{left} vs {right}",
            "sharpe_diff": obs_sharpe_diff,
            "bootstrap_p_value_sharpe": p_sharpe,
            "median_return_diff": obs_median_diff,
            "bootstrap_p_value_median": p_median,
            "n_months": int(m),
            "block_size": int(block_size),
        })

    return pd.DataFrame(rows)






def plot_correlation_heatmap(returns: pd.DataFrame, outfile: Path) -> None:
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Monthly return correlation heatmap")
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_cumulative_wealth(portfolios: pd.DataFrame, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for strategy, g in portfolios.groupby("strategy"):
        s = g.sort_values("date")
        wealth = (1.0 + s["return"]).cumprod()
        ax.plot(s["date"], wealth, label=strategy, linewidth=2 if strategy in {"EqualWeight", "SPY"} else 1.5)
    ax.set_title("Cumulative wealth of model-based and benchmark portfolios (gross)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_forecast_metric_bars(metrics: pd.DataFrame, outfile: Path) -> None:
    m = metrics.copy().set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    m["RMSE"].plot(kind="bar", ax=axes[0])
    axes[0].set_title("RMSE by model")
    m["MAE"].plot(kind="bar", ax=axes[1])
    axes[1].set_title("MAE by model")
    m["DirectionalAccuracy"].plot(kind="bar", ax=axes[2])
    axes[2].set_title("Directional accuracy")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_performance_heatmap(perf: pd.DataFrame, outfile: Path) -> None:
    cols = ["CAGR", "AnnualVol", "Sharpe", "Sortino", "MaxDrawdown", "Calmar", "FinalWealth"]
    df = perf.set_index("Strategy")[cols].copy()
    z = (df - df.mean()) / df.std(ddof=0)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(df))))
    im = ax.imshow(z.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title("Strategy performance heatmap (z-scored values)")
    for i in range(len(df.index)):
        for j in range(len(cols)):
            ax.text(j, i, f"{df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_feature_importance(fi: pd.DataFrame, outfile: Path, model_preference: str = "RF") -> None:
    if fi.empty:
        return
    tmp = fi.copy()
    if model_preference in tmp["model"].unique():
        tmp = tmp[tmp["model"] == model_preference]
    latest_date = tmp["date"].max()
    top = (
        tmp[tmp["date"] == latest_date]
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["feature"][::-1], top["importance"][::-1])
    ax.set_title(f"Top feature importances ({model_preference}, {pd.to_datetime(latest_date).date()})")
    ax.set_xlabel("Average importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_selection_frequency(selection: pd.DataFrame, outfile: Path) -> None:
    """
    selection: columns [ticker, model, count]
    """
    pivot = selection.pivot(index="ticker", columns="model", values="count").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(pivot.index))
    for model in pivot.columns:
        ax.bar(pivot.index, pivot[model], bottom=bottom, label=model)
        bottom += pivot[model].values
    ax.set_title("Sector selection frequency in Top-K portfolios")
    ax.set_ylabel("Number of selection months")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)






def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce the empirical analysis for the term paper.")
    p.add_argument("--start", default=DEFAULT_START_DATE, help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", default=DEFAULT_END_DATE, help="End date (YYYY-MM-DD).")
    p.add_argument("--train-years", type=int, default=DEFAULT_TRAIN_YEARS, help="Initial expanding-window training size in years.")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K sectors selected each month.")
    p.add_argument("--rf", type=float, default=DEFAULT_RF_ANNUAL, help="Annual risk-free rate (for Sharpe/Sortino).")
    p.add_argument("--tcost-bps", type=float, default=DEFAULT_TCOST_BPS, help="Turnover cost (bps) applied to turnover share.")
    p.add_argument("--weighting", choices=["equal", "invvol", "riskparity"], default="equal", help="Weighting rule inside Top-K.")
    p.add_argument("--window", choices=["expanding", "rolling"], default="expanding", help="Training window design for OOS forecasts.")
    p.add_argument("--rolling-months", type=int, default=36, help="Rolling training-window length in months when --window rolling is used.")
    p.add_argument("--outdir", default="ml_asset_management_outputs", help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    figdir, tabdir = ensure_dirs(outdir)


    print("Downloading data (yfinance)...")
    prices = download_monthly_prices(list(TICKERS.keys()), args.start, args.end)
    rets = monthly_returns(prices)


    print("Engineering features...")
    features = make_asset_features(rets)

    print("Running expanding-window forecasts...")
    fr = rolling_forecasts(features, train_years=args.train_years, window=args.window, rolling_months=args.rolling_months)


    print("Constructing portfolios...")
    portfolios_gross, weights = build_portfolios(fr.forecasts, rets, top_k=args.top_k, weighting=args.weighting)


    asset_cols = [c for c in rets.columns if c != "SPY"]
    turnover_rows = []
    for strat, g in weights.groupby("strategy"):
        w = g.sort_values("date")[asset_cols]
        turn = compute_turnover(w)
        turnover_rows.append({"Strategy": strat, "AverageMonthlyTurnover": float(turn.mean())})
    turnover = pd.DataFrame(turnover_rows).sort_values("AverageMonthlyTurnover", ascending=False).reset_index(drop=True)


    tcost = args.tcost_bps / 10000.0
    portfolios_net = portfolios_gross.copy()
    portfolios_net = portfolios_net.merge(
        weights[["date", "strategy"] + asset_cols],
        on=["date", "strategy"],
        how="left",
    )

    net_rows = []
    for strat, g in portfolios_net.groupby("strategy"):
        g = g.sort_values("date").copy()
        w = g[asset_cols].fillna(0.0)
        turn = compute_turnover(w)
        g["turnover"] = turn.values
        g["return_net"] = g["return"] - tcost * g["turnover"]
        net_rows.append(g[["date", "strategy", "return_net"]])
    portfolios_net_long = pd.concat(net_rows, ignore_index=True).rename(columns={"return_net": "return"})


    perf_gross = performance_table(portfolios_gross, rf_annual=args.rf)
    perf_net = performance_table(portfolios_net_long, rf_annual=args.rf)
    dist_gross = strategy_return_distribution(portfolios_gross)

    bootstrap_tests = block_bootstrap_portfolio_tests(
        portfolios_gross,
        comparisons=[("GB", f"MomentumTop{args.top_k}"), ("RF", f"MomentumTop{args.top_k}"), ("HistMean", f"MomentumTop{args.top_k}"), ("GB", "SPY")],
        rf_annual=args.rf,
        n_boot=2000,
        block_size=6,
        seed=42,
    )


    print("Computing selection frequency...")
    asset_universe = [t for t in fr.forecasts["ticker"].unique() if t != "SPY"]
    selection_rows = []
    for model in sorted(fr.forecasts["model"].unique()):
        for dt, g in fr.forecasts[(fr.forecasts["model"] == model) & (fr.forecasts["ticker"].isin(asset_universe))].groupby("date"):
            chosen = g.sort_values("forecast", ascending=False).head(args.top_k)["ticker"].tolist()
            for t in chosen:
                selection_rows.append({"model": model, "ticker": t})
    selection = pd.DataFrame(selection_rows).groupby(["ticker", "model"]).size().reset_index(name="count")


    cal_year = calendar_year_returns(portfolios_gross)
    yr_sel = year_end_selections(fr.forecasts, top_k=args.top_k)


    desc_returns = descriptive_statistics(rets, rf_annual=args.rf)
    desc_targets = descriptive_statistics(pd.DataFrame({k: v["target"] for k, v in features.items()}), rf_annual=args.rf)


    feat_desc_rows = []
    for tkr, df in features.items():
        tmp = descriptive_statistics(df.drop(columns=["target"]), rf_annual=args.rf)
        tmp.insert(0, "Ticker", tkr)
        feat_desc_rows.append(tmp)
    desc_features = pd.concat(feat_desc_rows, ignore_index=True)


    print("Saving tables...")
    rets.to_csv(tabdir / "monthly_returns.csv")
    fr.forecasts.to_csv(tabdir / "model_forecasts.csv", index=False)
    fr.metrics.to_csv(tabdir / "forecast_metrics.csv", index=False)
    fr.dm_tests.to_csv(tabdir / "forecast_metrics_dm.csv", index=False)
    fr.cw_tests.to_csv(tabdir / "forecast_metrics_clark_west.csv", index=False)
    portfolios_gross.to_csv(tabdir / "portfolio_returns_gross.csv", index=False)
    portfolios_net_long.to_csv(tabdir / "portfolio_returns_net.csv", index=False)
    perf_gross.to_csv(tabdir / "portfolio_performance_gross.csv", index=False)
    perf_net.to_csv(tabdir / "portfolio_performance_net.csv", index=False)
    turnover.to_csv(tabdir / "turnover_summary.csv", index=False)
    dist_gross.to_csv(tabdir / "strategy_return_distribution.csv", index=False)
    bootstrap_tests.to_csv(tabdir / "portfolio_bootstrap_tests.csv", index=False)
    selection.to_csv(tabdir / "selection_frequency.csv", index=False)
    yr_sel.to_csv(tabdir / "year_end_selections.csv", index=False)
    cal_year.to_csv(tabdir / "calendar_year_returns.csv", index=False)
    desc_returns.to_csv(tabdir / "descriptive_statistics_monthly_returns.csv", index=False)
    desc_targets.to_csv(tabdir / "descriptive_statistics_targets.csv", index=False)
    desc_features.to_csv(tabdir / "descriptive_statistics_features.csv", index=False)

    if not fr.feature_importance.empty:
        fr.feature_importance.to_csv(tabdir / "feature_importance_raw.csv", index=False)


    print("Creating figures...")
    plot_correlation_heatmap(rets, figdir / "01_correlation_heatmap.png")
    plot_cumulative_wealth(portfolios_gross, figdir / "02_cumulative_wealth_gross.png")
    plot_forecast_metric_bars(fr.metrics, figdir / "03_forecast_metric_bars.png")
    plot_performance_heatmap(perf_gross, figdir / "04_strategy_performance_heatmap.png")
    plot_feature_importance(fr.feature_importance, figdir / "05_feature_importance.png", model_preference="RF")
    plot_selection_frequency(selection, figdir / "06_sector_selection_frequency.png")

    print("Done.")
    print(f"Tables saved to: {tabdir.resolve()}")
    print(f"Figures saved to: {figdir.resolve()}")


if __name__ == "__main__":
    main()
