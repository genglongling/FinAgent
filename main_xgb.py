#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

#####################################################################
# User Configuration
#####################################################################
TARGET_SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "TSLA",  # Tesla
    "GOOG",  # Alphabet
    "META",  # Meta
    "JPM",   # JP Morgan Chase & Co.
    "BX",    # Blackstone Inc.
    "GS",    # Goldman Sachs Group, Inc. (The)
    "RJF",   # Raymond James Financial, Inc.
    "V",     # Visa Inc.
    "MA",    # Mastercard Incorporated
    "NFLX",  # Netflix, Inc.
    "TMUS",  # T-Mobile US, Inc.
    "DIS",   # The Walt Disney Company
    "VZ",    # Verizon Communications Inc.
    "LYV",   # Live Nation Entertainment, Inc.
    "FOX",   # Fox Corporation
]

STOCK_CSV_PATH = os.path.join("sp500_stocks", "sp500_stocks.csv")

# Number of past days to include as features
LOOKBACK_DAYS = 3

# Ratio of training data vs. total
TRAIN_SPLIT_RATIO = 0.8

#####################################################################
# 1. LOAD DATA WITH NaN HANDLING
#####################################################################
def load_full_data():
    """
    Loads the CSV file containing all S&P 500 stock data.
    Returns a DataFrame.
    If the file is not found, the script exits.
    """
    if not os.path.exists(STOCK_CSV_PATH):
        sys.exit(f"[ERROR] CSV file not found at: {STOCK_CSV_PATH}")

    print(f"[INFO] Loading dataset from: {STOCK_CSV_PATH}")
    df = pd.read_csv(STOCK_CSV_PATH)

    # Ensure Date column is in datetime format
    if 'Date' in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # Sort by Symbol and Date
    df.sort_values(by=["Symbol", "Date"], inplace=True)

    # Fill missing values in Close prices with forward-fill and backward-fill
    df["Close"] = df.groupby("Symbol")["Close"].ffill().bfill()

    print(f"[INFO] DataFrame shape: {df.shape}")
    print(df.head(3))
    return df

#####################################################################
# 2. FILTER BY SYMBOL
#####################################################################
def filter_by_symbol(df, symbol):
    """
    Returns only the rows for the specified stock symbol.
    """
    df_symbol = df[df['Symbol'] == symbol].copy()
    if df_symbol.empty:
        print(f"[WARNING] No data found for {symbol}.")
    else:
        print(f"[INFO] {symbol}: {len(df_symbol)} rows found.")
    return df_symbol

#####################################################################
# 3. PREPROCESS DATA & CREATE MULTI-DAY FEATURES
#####################################################################
def preprocess_data_xgb(df_symbol, lookback_days=LOOKBACK_DAYS):
    """
    Prepares the DataFrame for XGBoost training:
      - Generates past days' closing prices as features.
      - Creates the target column (next-day close).
      - Handles missing values.
    """
    df_symbol = df_symbol.copy()

    # Create lagged features for past LOOKBACK_DAYS closes
    for i in range(1, lookback_days + 1):
        df_symbol[f"Close_Lag_{i}"] = df_symbol["Close"].shift(i)

    # Target variable: Next-day Close
    df_symbol["Target"] = df_symbol["Close"].shift(-1)

    # Drop rows with NaN values (caused by shifting)
    df_symbol.dropna(inplace=True)

    # Ensure Date is sorted correctly
    df_symbol.sort_values(by="Date", inplace=True)

    return df_symbol

#####################################################################
# 4. TIME-BASED TRAIN-VALIDATION SPLIT
#####################################################################
def time_based_split(df, date_col="Date", split_ratio=TRAIN_SPLIT_RATIO):
    """
    Splits df into train/val based on chronological order (80/20 by default).
    """
    df = df.sort_values(by=date_col)
    unique_dates = df[date_col].unique()
    cutoff_index = int(len(unique_dates) * split_ratio)
    cutoff_date = unique_dates[cutoff_index]

    df_train = df[df[date_col] <= cutoff_date]
    df_val = df[df[date_col] > cutoff_date]

    return df_train, df_val, cutoff_date

#####################################################################
# 5. TRAIN XGBOOST WITH MULTI-DAY CLOSE PRICES AS FEATURES
#####################################################################
def train_xgb(df_train, lookback_days=LOOKBACK_DAYS):
    """
    Trains an XGBoost model using multiple past days of Close as features.
    """
    feature_cols = [f"Close_Lag_{i}" for i in range(1, lookback_days + 1)]

    X_train = df_train[feature_cols].values
    y_train = df_train["Target"].values

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

#####################################################################
# 6. PLOT TRAIN & VAL PREDICTIONS IN ONE FIGURE
#####################################################################
def plot_train_val(
    df_train, df_val, y_pred_train, y_pred_val, cutoff_date, symbol
):
    """
    Combine train & val into one DataFrame. Plot Real vs. Predicted
    next-day Close on the same figure. A vertical line shows the
    train/validation split date.
    """
    df_train = df_train.copy()
    df_train["Real"] = df_train["Target"]
    df_train["Predicted"] = y_pred_train
    df_train["Set"] = "Train"

    df_val = df_val.copy()
    df_val["Real"] = df_val["Target"]
    df_val["Predicted"] = y_pred_val
    df_val["Set"] = "Validation"

    df_plot = pd.concat([df_train, df_val], ignore_index=True)
    df_plot.sort_values("Date", inplace=True)

    plt.figure(figsize=(10, 6))
    plt.title(f"Train & Validation Next-Day Close\nSymbol: {symbol}")
    plt.plot(df_plot["Date"], df_plot["Real"], label="Real Next-Day Close", color="blue")
    plt.plot(df_plot["Date"], df_plot["Predicted"], label="Predicted Next-Day Close", color="orange")
    plt.axvline(x=cutoff_date, color="red", linestyle="--", label="Train/Val Split")

    plt.xlabel("Date")
    plt.ylabel("Next-Day Close Price")
    plt.legend()
    plt.tight_layout()

    # Ensure output directory exists
    output_dir = "output_xgb"
    os.makedirs(output_dir, exist_ok=True)

    out_file = os.path.join(output_dir, f"{symbol}_train_val.png")
    plt.savefig(out_file)
    plt.close()
    print(f"  [INFO] Combined plot saved to: {out_file}")

def save_performance_metrics(symbol, train_mse, val_mse, model_type="xgb"):
    """
    Save performance metrics to a CSV file.
    
    Args:
        symbol (str): Stock symbol
        train_mse (float): Training MSE
        val_mse (float): Validation MSE
        model_type (str): Type of model used
    """
    metrics_df = pd.DataFrame({
        'Symbol': [symbol],
        'Model': [model_type],
        'Train_MSE': [train_mse],
        'Val_MSE': [val_mse],
        'Train_RMSE': [np.sqrt(train_mse)],
        'Val_RMSE': [np.sqrt(val_mse)]
    })
    
    output_dir = "performance_metrics"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "model_performance.csv")
    
    # If file exists, append to it; otherwise create new
    if os.path.exists(output_file):
        metrics_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(output_file, index=False)
    
    print(f"  [INFO] Performance metrics saved to: {output_file}")

#####################################################################
# 7. MAIN
#####################################################################
def main():
    # 1. Load dataset
    df = load_full_data()

    # 2. Loop through stock symbols
    for symbol in TARGET_SYMBOLS:
        print(f"\n[INFO] Processing symbol: {symbol}")

        df_symbol = filter_by_symbol(df, symbol)
        if df_symbol.empty:
            continue

        # 3. Prepare data for XGBoost
        df_xgb = preprocess_data_xgb(df_symbol)

        # Check if we have enough data
        if len(df_xgb) < 50:
            print(f"[WARNING] {symbol}: insufficient data after preprocessing.")
            continue
        if df_xgb.shape[0] < 2:
            print(f"[ERROR] {symbol}: Not enough valid data for forecasting.")
            continue

        # 4. Split data into train/validation sets
        df_train, df_val, cutoff_date = time_based_split(df_xgb, "Date", TRAIN_SPLIT_RATIO)

        print(f"  [INFO] {symbol} -> Train size: {len(df_train)}, Val size: {len(df_val)}")

        # 5. Train XGBoost model
        model = train_xgb(df_train)

        # 6. Predict on training & validation sets
        feature_cols = [f"Close_Lag_{i}" for i in range(1, LOOKBACK_DAYS + 1)]
        y_pred_train = model.predict(df_train[feature_cols].values)
        y_pred_val = model.predict(df_val[feature_cols].values)

        # 7. Compute MSE
        train_mse = mean_squared_error(df_train["Target"], y_pred_train)
        val_mse = mean_squared_error(df_val["Target"], y_pred_val)
        print(f"  [TRAIN] MSE: {train_mse:.4f}")
        print(f"  [VAL]   MSE: {val_mse:.4f}")

        # Save performance metrics
        save_performance_metrics(symbol, train_mse, val_mse)

        # 8. Plot results
        plot_train_val(df_train, df_val, y_pred_train, y_pred_val, cutoff_date, symbol)

if __name__ == "__main__":
    main()