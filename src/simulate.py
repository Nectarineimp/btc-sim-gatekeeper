#!/usr/bin/env python3
"""
GateKeeper: Forward 12-Month Stochastic Forecast Simulator
Architecture: Coupled Macro-Liquidity Gated SDE with 8-Month Institutional Mandate Rebalancing
Schema: date (MM/DD/YYYY), close (float), log_price (float), log_return (float)
"""

import argparse
import sys
import numpy as np
import pandas as pd
from datetime import timedelta


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run GateKeeper Forward 12-Month Stochastic Forecast Simulation."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="btc_daily_price.csv",
        help="Path to input btc_daily_price.csv historical data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gatekeeper_forecast.csv",
        help="Path to export the quantile forecast CSV.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of Monte Carlo path iterations (default: 10000).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Forward forecast horizon in days (default: 365).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for path reproducibility (default: 42).",
    )
    return parser.parse_args()


def run_simulation(data_path, output_path, n_sims, forecast_days, seed):
    np.random.seed(seed)

    # 1. Ingest historical price data using explicit format and column mapping
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading input data from {data_path}: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {"date", "close"}
    if not required_cols.issubset(df.columns):
        print(
            f"Input CSV missing required columns. Expected at least {required_cols}, found: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse MM/DD/YYYY format
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)

    # Clean numeric close
    s0 = float(df["close"].iloc[-1])
    start_date = df["date"].iloc[-1]

    # Genesis anchor for power-law trajectory (Jan 3, 2009)
    genesis_date = pd.to_datetime("2009-01-03")
    t0_days = (start_date - genesis_date).days

    # ------------------------------------------------------------
    # 2. Econometric Parameters
    # ------------------------------------------------------------
    # Dynamic power-law trajectory anchored to spot start date:
    # S_PL(t) = S_0 * (t / t_0)^beta_pl
    beta_pl = 5.67
    sigma_base = 0.42          # Compressed institutional baseline annualized volatility (42%)
    dt = 1.0 / 365.0
    sqrt_dt = np.sqrt(dt)

    # ------------------------------------------------------------
    # Institutional Mandate Rebalancing Parameters
    # ------------------------------------------------------------
    rebalance_period = 240.0   # 8-month harmonic institutional rebalancing mode (~240 days)
    kappa_rebalance = 1.20     # Mean-reverting restoration elasticity
    upper_log_bound = 0.30     # Cycle performance upper ceiling (+0.30 log units)
    lower_log_bound = -0.50    # Cycle performance lower floor (-0.50 log units)

    # ------------------------------------------------------------
    # Macro Liquidity Gate Parameters
    # ------------------------------------------------------------
    net_liquidity_growth = 0.015  # Baseline neutral liquidity regime
    tau = 0.020                  # Liquidity activation threshold

    # ------------------------------------------------------------
    # 3. Path Pre-allocation (days + 1, n_sims)
    # ------------------------------------------------------------
    prices = np.empty((forecast_days + 1, n_sims), dtype=np.float64)
    prices[0, :] = s0

    # ------------------------------------------------------------
    # 4. Simulation Engine (Vectorized across all paths per daily time step)
    # ------------------------------------------------------------
    for step in range(1, forecast_days + 1):
        current_day = t0_days + step

        # Dynamically anchored Power-Law benchmark
        pl_price = s0 * ((current_day / t0_days) ** beta_pl)

        # 1. Compute Log-residual deviation FIRST
        log_deviation = np.log(prices[step - 1, :] / pl_price)

        # 2. 8-month institutional review cycle modulation: Phi(t)
        phi_t = 1.0 + 0.5 * np.cos(2.0 * np.pi * step / rebalance_period)
        restoring_force = -kappa_rebalance * phi_t * log_deviation

        # 3. State-dependent drift calculations:
        # (a) Liquidity gate
        gated_mu = 0.01 + 0.40 * max(0.0, net_liquidity_growth - tau)

        # (b) Continuous cubic restoring barrier enforcing [-0.50, +0.30] corridor
        upper_overshoot = np.maximum(0.0, log_deviation - upper_log_bound)
        lower_undershoot = np.maximum(0.0, lower_log_bound - log_deviation)
        
        upper_penalty = -2.0 * upper_overshoot - 8.0 * (upper_overshoot ** 3)
        lower_support = 2.0 * lower_undershoot + 8.0 * (lower_undershoot ** 3)

        # Composite instantaneous annualized drift
        mu_t = gated_mu + restoring_force + upper_penalty + lower_support
        
        # Clamp annualized drift to realistic institutional bounds [-30%, +35%]
        mu_t = np.clip(mu_t, -0.30, 0.35)

        # Volatility modulation: compressed in corridor, widens slightly near extremes
        sigma_t = sigma_base * (1.0 + 0.25 * np.clip(np.abs(log_deviation), 0.0, 1.0))

        # Euler-Maruyama stochastic step with log-exponent safety clipping
        z = np.random.standard_normal(n_sims)
        log_increment = (mu_t - 0.5 * (sigma_t ** 2)) * dt + sigma_t * sqrt_dt * z
        log_increment = np.clip(log_increment, -0.15, 0.15)  # Daily move clamped to realistic +/- 15%
        
        prices[step, :] = prices[step - 1, :] * np.exp(log_increment)

    # ------------------------------------------------------------
    # 5. Monthly Low/High Quantile Aggregation
    # ------------------------------------------------------------
    dates = [start_date + timedelta(days=i) for i in range(forecast_days + 1)]
    dates_pd = pd.to_datetime(dates)
    month_periods = dates_pd.to_period("M")
    unique_months = month_periods.unique()

    # Exclude initial partial start month to align forward 12-month horizon
    forecast_months = unique_months[1:13] if len(unique_months) > 12 else unique_months[1:]

    records = []
    for m in forecast_months:
        mask = (month_periods == m)
        # prices shape: (days + 1, n_sims) -> slice month days
        month_prices = prices[mask, :]

        # Path-wise monthly extrema across all simulated runs
        path_lows = np.min(month_prices, axis=0)
        path_highs = np.max(month_prices, axis=0)

        records.append({
            "Month": str(m),
            "Low_p05": np.percentile(path_lows, 5),
            "Low_p16": np.percentile(path_lows, 16),
            "Low_p50": np.percentile(path_lows, 50),
            "Low_p84": np.percentile(path_lows, 84),
            "Low_p95": np.percentile(path_lows, 95),
            "High_p05": np.percentile(path_highs, 5),
            "High_p16": np.percentile(path_highs, 16),
            "High_p50": np.percentile(path_highs, 50),
            "High_p84": np.percentile(path_highs, 84),
            "High_p95": np.percentile(path_highs, 95),
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_path, index=False)
    print(f"GateKeeper forecast generated successfully: {output_path} ({n_sims} iterations across {len(records)} months)")


def main():
    args = parse_arguments()
    run_simulation(
        data_path=args.data,
        output_path=args.output,
        n_sims=args.iterations,
        forecast_days=args.days,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
