#!/usr/bin/env python3

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from magentic import prompt

TARGET_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "TSLA", "GOOG", "META",
    "JPM",  "BX",   "GS",   "RJF",  "V",    "MA",
    "NFLX", "TMUS", "DIS",  "VZ",   "LYV",  "FOX",
]

STOCK_CSV_PATH = os.path.join("sp500_stocks", "sp500_stocks.csv")
LOOKBACK_DAYS = 5

###############################################################################
# 1) LOAD DATA
###############################################################################
def load_full_data():
    if not os.path.exists(STOCK_CSV_PATH):
        sys.exit(f"[ERROR] CSV file not found at: {STOCK_CSV_PATH}")
    print(f"[INFO] Loading dataset from: {STOCK_CSV_PATH}")
    df = pd.read_csv(STOCK_CSV_PATH)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by=["Symbol","Date"], inplace=True)

    # Fill missing Close forward/backward
    df["Close"] = df.groupby("Symbol")["Close"].ffill().bfill()
    return df

def filter_by_symbol(df, symbol):
    df_symbol = df[df["Symbol"] == symbol].copy()
    if df_symbol.empty:
        print(f"[WARNING] No data for {symbol}")
    else:
        print(f"[INFO] {symbol}: {len(df_symbol)} rows found.")
    return df_symbol

###############################################################################
# 2) ADD TECHNICAL FEATURES (PAST DATA ONLY)
###############################################################################
def add_technical_features(df):
    """
    Compute rolling indicators but shift(1) so each row sees only past data.
    """
    df = df.copy()

    # --- SHIFT ROLLING AVERAGES BY 1 DAY ---
    df["SMA_10"] = df["Close"].rolling(window=10).mean().shift(1)
    df["SMA_50"] = df["Close"].rolling(window=50).mean().shift(1)
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean().shift(1)

    df["SMA_20"] = df["Close"].rolling(window=20).mean().shift(1)
    df["STD_20"] = df["Close"].rolling(window=20).std().shift(1)
    df["Bollinger_Upper"] = df["SMA_20"] + 2 * df["STD_20"]
    df["Bollinger_Lower"] = df["SMA_20"] - 2 * df["STD_20"]
    df["Bollinger_Width"] = (df["Bollinger_Upper"] - df["Bollinger_Lower"]) / df["SMA_20"]

    # --- SHIFT(1) FOR MACD CALCS ---
    ema12 = df["Close"].ewm(span=12, adjust=False).mean().shift(1)
    ema26 = df["Close"].ewm(span=26, adjust=False).mean().shift(1)
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # --- SHIFT(1) FOR ATR & RSI ---
    df["Prev_Close"] = df["Close"].shift(1)
    df["High_Low"] = df["High"] - df["Low"]
    df["High_PrevClose"] = (df["High"] - df["Prev_Close"]).abs()
    df["Low_PrevClose"] = (df["Low"] - df["Prev_Close"]).abs()
    df["True_Range"] = df[["High_Low", "High_PrevClose", "Low_PrevClose"]].max(axis=1)
    df["ATR_14"] = df["True_Range"].rolling(window=14).mean().shift(1)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean().shift(1)
    avg_loss = loss.rolling(window=14).mean().shift(1)
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100/(1+rs))

    # --- SHIFT(1) FOR OBV ---
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum().shift(1)

    df.drop(columns=["Prev_Close","High_Low","High_PrevClose","Low_PrevClose"], inplace=True)
    return df

def add_rolling_features(df, window=7):
    """
    Rolling High/Low/Volume, also shifted by 1 day
    so we only use past data.
    """
    lag_features = ["High", "Low", "Volume"]
    df_rolled = df[lag_features].rolling(window=window, min_periods=0).mean().shift(1)
    df_rolled = df_rolled.astype(np.float32)
    df_rolled.columns = [f"{col}_{window}d" for col in lag_features]
    return df_rolled

###############################################################################
# 3) PREPROCESS
###############################################################################
def preprocess_data_for_llm(df_symbol):
    """
    1) Add technical indicators / rolling features using past data only.
    2) Create lag features & next-day Target.
    3) Drop rows with missing values.
    4) Sort, then filter by date after 2022.
    """
    df_symbol = add_technical_features(df_symbol)
    df_rolling = add_rolling_features(df_symbol, window=7)
    df_symbol = pd.concat([df_symbol.reset_index(drop=True),
                           df_rolling.reset_index(drop=True)], axis=1)
    
    # Lags => strictly references past Close
    for i in range(1, LOOKBACK_DAYS + 1):
        df_symbol[f"Close_Lag_{i}"] = df_symbol["Close"].shift(i)

    # Next-day close as target
    df_symbol["Target"] = df_symbol["Close"].shift(-1)

    # Drop rows that still have missing features
    df_symbol.dropna(inplace=True)
    df_symbol.sort_values(by="Date", inplace=True)

    # Filter after 2022 for evaluation
    df_symbol = df_symbol[df_symbol["Date"] >= "2022-01-01"].copy()
    return df_symbol

###############################################################################
# 4) LLM PROMPT
###############################################################################
@prompt(
    """
    You are a stock market forecasting assistant with extensive financial knowledge.
    The following numeric features are known for a particular day:
    
    {feature_description}
    
    Predict the NEXT day's closing price as a FLOAT ONLY (e.g., 145.32).
    No disclaimers or extra text.
    """
)
def llm_predict_next_day(feature_description: str) -> str:
    """LLM prompt function. We'll parse its output as float."""
    ...

###############################################################################
# 5) ROW-BY-ROW PREDICTION
###############################################################################
def predict_llm_with_skip(df, feature_cols, max_retries=3, sleep_sec=1.0):
    """
    For each row, if any feature is missing => skip (np.nan).
    Otherwise, call LLM. If invalid float => retry up to max_retries.
    If still invalid => skip.
    """
    predictions = []
    for _, row in df.iterrows():
        if row[feature_cols].isnull().any():
            predictions.append(np.nan)
            continue

        # Create text from feature cols
        desc_text = "\n".join(f"{col}={row[col]:.4f}" for col in feature_cols)

        # Attempt LLM calls
        pred_float = np.nan
        for attempt in range(max_retries):
            raw_resp = llm_predict_next_day(desc_text)
            try:
                pred_float = float(raw_resp.strip())
                break
            except ValueError:
                print(f"[WARNING] LLM invalid float attempt {attempt+1}: {raw_resp!r}")
                time.sleep(sleep_sec)

        if np.isnan(pred_float):
            print(f"[WARNING] LLM gave no valid numeric after {max_retries} attempts => skip.")
        predictions.append(pred_float)

    return np.array(predictions)

###############################################################################
# 6) PLOT
###############################################################################
def plot_predictions(df, y_pred, symbol, start_date="2022-01-01"):
    df = df.copy()
    df["Predicted"] = y_pred
    df = df[df["Date"] >= start_date].copy()
    df.sort_values("Date", inplace=True)

    plt.figure(figsize=(10, 6))
    plt.title(f"LLM Next-Day Close (Past-Only Indicators)\nSymbol: {symbol}")
    plt.plot(df["Date"], df["Target"], label="Actual Next-Day Close", color="blue")
    plt.plot(df["Date"], df["Predicted"], label="LLM Prediction", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Next-Day Close Price")
    plt.legend()
    plt.tight_layout()

    out_dir = "output_llm_local"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{symbol}_llm_local.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[INFO] Plot saved to {out_file}")

###############################################################################
# 7) MAIN
###############################################################################
def main():
    df = load_full_data()
    # Optionally limit data to 2024
    df = df[df["Date"] <= "2024-12-31"]

    for symbol in TARGET_SYMBOLS:
        df_symbol = filter_by_symbol(df, symbol)
        if df_symbol.empty:
            continue

        df_llm = preprocess_data_for_llm(df_symbol)
        if df_llm.empty:
            print(f"[WARNING] {symbol}: after processing, no data.")
            continue

        # Feature columns
        feature_cols = (
            [f"Close_Lag_{i}" for i in range(1, LOOKBACK_DAYS+1)]
            + ["High_7d","Low_7d","Volume_7d",
               "SMA_10","SMA_50","EMA_10","Bollinger_Width",
               "MACD","RSI_14","OBV","ATR_14"]
        )
        # Only use columns that exist
        feature_cols = [c for c in feature_cols if c in df_llm.columns]

        # Predict row-by-row
        y_pred = predict_llm_with_skip(df_llm, feature_cols, max_retries=3, sleep_sec=1.0)

        # Evaluate
        mask = (~np.isnan(y_pred)) & (~df_llm["Target"].isnull())
        if not mask.any():
            print(f"[INFO] {symbol}: no valid predictions. All skipped.")
            continue

        mse = mean_squared_error(df_llm.loc[mask, "Target"], y_pred[mask])
        rmse = np.sqrt(mse)
        print(f"\n[INFO] {symbol} => LLM MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        # Plot
        plot_predictions(df_llm, y_pred, symbol, start_date="2022-01-01")

    print("\n[INFO] Done with LLM predictions (past-only indicators)!\n")

if __name__ == "__main__":
    main()
