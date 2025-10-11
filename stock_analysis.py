"""
stock_analysis_project.py

Presentation-ready stock analysis project (>100 lines).
Features:
- multi-ticker download via yfinance (or CSV fallback)
- cleaning, filtering, conversion (daily -> monthly)
- returns, cumulative returns, annualized return, volatility, Sharpe ratio
- drawdown and correlation analysis
- saves plots and a simple markdown report

Run:
    python stock_analysis_project.py
"""

import os
import sys
import warnings
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Suppress yfinance FutureWarning for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


# Configuration

TICKERS = ["AAPL", "MSFT", "GOOGL"]
START_DATE = "2019-10-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
OUTPUT_DIR = "output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.md")
RISK_FREE_RATE = 0.01

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# -------------------------
# Utilities
# -------------------------
def save_fig(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved: {path}")


def ensure_df_index_date(df: pd.DataFrame, date_col: str = "Date"):
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    return df


# -------------------------
# Data Loading
# -------------------------
def download_data(tickers, start="2019-01-01", end=None) -> Dict[str, pd.DataFrame]:
    """
    Download daily data per ticker (one ticker per request) to avoid multi-ticker columns complexity.
    Returns a dict ticker -> DataFrame.
    """
    print("Downloading stock data...")
    data = {}
    for t in tickers:
        print(f"\nDownloading {t} from {start} to {end} ...")
        # Force auto_adjust=False so we keep both Close and Adj Close
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)

        if df.empty:
            print(f"  ⚠️ Warning: {t} returned empty data. Check ticker or connectivity.")
            continue

        # Show columns for debugging
        print(f"  raw columns: {list(df.columns[:10])}")

        # If MultiIndex (metric, ticker), flatten to metric names (I downloaded one ticker at a time so this keeps metric names)
        if isinstance(df.columns, pd.MultiIndex):
            # col example: ('Adj Close','AAPL') -> use first level 'Adj Close'
            df.columns = [col[0] for col in df.columns]
            print(f"  flattened columns: {list(df.columns[:10])}")

        # Reset index and ensure datetime index
        df = df.reset_index()
        df = ensure_df_index_date(df, "Date")

        data[t] = df
    return data


def read_csv_fallback(ticker: str, path: str) -> Optional[pd.DataFrame]:
    """Load from CSV if needed (expects a Date column)."""
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = ensure_df_index_date(df, "Date")
        print(f"Loaded fallback CSV for {ticker} from {path}")
        return df
    except Exception as e:
        print(f"CSV fallback failed for {ticker}: {e}")
        return None


# Cleaning and Preprocess

def clean_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean and standardize stock data for a specific ticker:
    - find appropriate price column (Adj Close preferred, then Close)
    - forward/back fill small gaps
    - ensure chronological sort
    """
    df = df.copy().sort_index()

    # Prefer 'Adj Close', then 'Close', then any column with 'close' substring, then numeric fallback
    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
        chosen = "Adj Close"
    elif "Close" in df.columns:
        df["Price"] = df["Close"]
        chosen = "Close"
    else:
        possible_cols = [c for c in df.columns if "close" in c.lower()]
        if possible_cols:
            df["Price"] = df[possible_cols[0]]
            chosen = possible_cols[0]
        else:
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
            if len(numeric_cols) == 0:
                raise ValueError(f"No numeric price column found for {ticker}. Columns: {list(df.columns)}")
            df["Price"] = df[numeric_cols[0]]
            chosen = numeric_cols[0]

    print(f"  [{ticker}] using '{chosen}' as Price column")

    # Fill small gaps and drop rows without price
    df["Price"] = df["Price"].ffill().bfill()
    df.dropna(subset=["Price"], inplace=True)
    df.sort_index(inplace=True)

    # Save cleaned CSV for reproducibility
    out_csv = os.path.join(DATA_DIR, f"{ticker}_cleaned.csv")
    df.to_csv(out_csv, index=True)
    print(f"  Saved cleaned data to {out_csv}")

    return df


# Analysis & Metrics

def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Daily Return"] = df["Price"].pct_change()
    df["Log Return"] = np.log(df["Price"] / df["Price"].shift(1))
    return df


def resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df_month = pd.DataFrame()
    df_month["first"] = df["Price"].resample("M").first()
    df_month["last"] = df["Price"].resample("M").last()
    df_month["mean"] = df["Price"].resample("M").mean()
    df_month.dropna(inplace=True)
    df_month["Monthly Return"] = df_month["last"].pct_change()
    return df_month


def compute_annualized_metrics(df: pd.DataFrame) -> dict:
    returns = df["Daily Return"].dropna()
    if returns.empty:
        return {}

    cumulative_return = (1 + returns).prod() - 1
    years = returns.shape[0] / 252.0
    annualized_return = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else np.nan
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe = (annualized_return - RISK_FREE_RATE) / annualized_vol if annualized_vol != 0 else np.nan

    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }


# Visualization

def plot_price_and_rolling(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Price"], label="Price")
    ax.plot(df.index, df["Price"].rolling(window=50).mean(), "--", label="50-day MA")
    ax.set_title(f"{ticker} Price & 50-day Rolling Mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    save_fig(fig, f"{ticker}_price_rolling.png")
    plt.close(fig)


def plot_monthly_returns(monthly_df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    monthly_df["Monthly Return"].plot(kind="bar", ax=ax)
    ax.set_title(f"{ticker} Monthly Returns")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly Return")
    plt.xticks(rotation=45)
    save_fig(fig, f"{ticker}_monthly_returns.png")
    plt.close(fig)


def plot_cumulative_returns(dfs: Dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(10, 6))
    for t, df in dfs.items():
        cum = (1 + df["Daily Return"].fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=t)
    ax.set_title("Cumulative Returns (Indexed to 1)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    save_fig(fig, "cumulative_returns.png")
    plt.close(fig)


def plot_correlation_heatmap(dfs: Dict[str, pd.DataFrame]):
    ret_df = pd.DataFrame({t: df["Log Return"] for t, df in dfs.items()}).dropna()
    corr = ret_df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="vlag", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Log Return Correlation")
    save_fig(fig, "correlation_heatmap.png")
    plt.close(fig)


def plot_drawdown(df: pd.DataFrame, ticker: str):
    cum = (1 + df["Daily Return"].fillna(0)).cumprod()
    drawdown = cum / cum.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.5)
    ax.set_title(f"{ticker} Drawdown")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    save_fig(fig, f"{ticker}_drawdown.png")
    plt.close(fig)


# Reporting

def generate_report(metrics: Dict[str, dict], tickers: List[str], out_path: str):
    lines = [
        "# Stock Analysis Report\n",
        f"**Tickers:** {', '.join(tickers)}\n",
        f"**Date range:** {START_DATE} → {END_DATE}\n",
        "## Summary Metrics\n",
    ]

    for t in tickers:
        m = metrics.get(t, {})
        if not m:
            lines.append(f"### {t}\nNo data available.\n")
            continue
        lines += [
            f"### {t}\n",
            f"- Annualized Return: {m['annualized_return']:.2%}\n",
            f"- Volatility: {m['annualized_volatility']:.2%}\n",
            f"- Sharpe Ratio: {m['sharpe_ratio']:.2f}\n",
            f"- Max Drawdown: {m['max_drawdown']:.2%}\n",
            f"![Price](plots/{t}_price_rolling.png)\n",
            f"![Monthly Returns](plots/{t}_monthly_returns.png)\n",
            f"![Drawdown](plots/{t}_drawdown.png)\n",
        ]

    lines.append("\n## Cross-Ticker Visuals\n")
    lines.append("![Cumulative Returns](plots/cumulative_returns.png)\n")
    lines.append("![Correlation Heatmap](plots/correlation_heatmap.png)\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {out_path}")


# Pipeline

def run_pipeline(tickers: List[str]):
    raw = download_data(tickers, START_DATE, END_DATE)
    processed, metrics = {}, {}

    for t, df in raw.items():
        df_clean = clean_data(df, t)
        df_returns = compute_daily_returns(df_clean)
        processed[t] = df_returns
        metrics[t] = compute_annualized_metrics(df_returns)

    if not processed:
        print("No data processed. Exiting.")
        return

    for t, df in processed.items():
        print(f"\nAnalyzing {t} — {len(df)} records")
        monthly = resample_to_monthly(df)
        plot_price_and_rolling(df, t)
        if not monthly.empty:
            plot_monthly_returns(monthly, t)
        plot_drawdown(df, t)
        m = metrics.get(t, {})
        if m:
            print(f"  Return={m['annualized_return']:.2%}, Sharpe={m['sharpe_ratio']:.2f}")

    plot_cumulative_returns(processed)
    plot_correlation_heatmap(processed)
    generate_report(metrics, list(processed.keys()), REPORT_PATH)
    print("\n Pipeline complete. Check the `output` folder.")


# Main Entry

if __name__ == "__main__":
    print("Starting stock analysis project...")
    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS
    run_pipeline(tickers)
