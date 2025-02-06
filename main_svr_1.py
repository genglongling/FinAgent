#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

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

# Number of past days to include as lag features for the Close price
LOOKBACK_DAYS = 5

# Ratio of training data vs. total
TRAIN_SPLIT_RATIO = 0.9

#####################################################################
# 1. LOAD DATA WITH NaN HANDLING
#####################################################################
def load_full_data():
    """
    Loads the CSV file containing all S&P 500 stock data.
    Expects columns including Date, Symbol, Adj Close, Close, High, Low, Open, Volume.
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

    # Fill missing values in Close (and you may consider doing similar for other columns)
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
# 3. ADD TECHNICAL FEATURES
#####################################################################
def add_technical_features(df):
    """
    Adds several technical indicators to the DataFrame using available columns:
    Adj Close, Close, High, Low, Open, Volume.
    
    Features added:
      - SMA_10, SMA_50: 10-day and 50-day Simple Moving Averages (using Close)
      - EMA_10: 10-day Exponential Moving Average (using Close)
      - Bollinger Bands (20-day): Upper, Lower, and Band Width
      - MACD and MACD_Signal: Moving Average Convergence Divergence
      - ATR_14: 14-day Average True Range
      - RSI_14: 14-day Relative Strength Index
      - OBV: On-Balance Volume
    """
    df = df.copy()

    # --- Moving Averages ---
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # --- Bollinger Bands (20-day) ---
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["STD_20"] = df["Close"].rolling(window=20).std()
    df["Bollinger_Upper"] = df["SMA_20"] + 2 * df["STD_20"]
    df["Bollinger_Lower"] = df["SMA_20"] - 2 * df["STD_20"]
    df["Bollinger_Width"] = (df["Bollinger_Upper"] - df["Bollinger_Lower"]) / df["SMA_20"]

    # --- MACD (Moving Average Convergence Divergence) ---
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # --- Average True Range (ATR) ---
    df["Prev_Close"] = df["Close"].shift(1)
    df["High_Low"] = df["High"] - df["Low"]
    df["High_PrevClose"] = np.abs(df["High"] - df["Prev_Close"])
    df["Low_PrevClose"] = np.abs(df["Low"] - df["Prev_Close"])
    df["True_Range"] = df[["High_Low", "High_PrevClose", "Low_PrevClose"]].max(axis=1)
    df["ATR_14"] = df["True_Range"].rolling(window=14).mean()

    # --- Relative Strength Index (RSI) ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # --- On-Balance Volume (OBV) ---
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # Drop intermediate columns used for calculations
    df.drop(columns=["Prev_Close", "High_Low", "High_PrevClose", "Low_PrevClose"], inplace=True)

    return df

#####################################################################
# 4. ADD ROLLING FEATURES (7-day mean for High, Low, Volume)
#####################################################################
def add_rolling_features(df, window=7):
    """
    Computes rolling means for the columns "High", "Low", and "Volume" over a window of 'window' days,
    shifts them by 1 (so the current row uses information from previous days only),
    and returns a DataFrame with new columns e.g., "High_7d", "Low_7d", "Volume_7d".
    """
    lag_features = ["High", "Low", "Volume"]
    df_rolled = df[lag_features].rolling(window=window, min_periods=0)
    df_mean = df_rolled.mean().shift(1)
    df_mean = df_mean.astype(np.float32)
    # Rename columns to indicate the window used
    df_mean.columns = [f"{col}_{window}d" for col in lag_features]
    return df_mean

#####################################################################
# 5. PREPROCESS DATA & CREATE FEATURES FOR SVR
#####################################################################
def preprocess_data_svr(df_symbol, lookback_days=LOOKBACK_DAYS):
    """
    Prepares the DataFrame for SVR training by:
      - Adding technical indicators.
      - Adding rolling features (7-day means for High, Low, Volume).
      - Generating past days' closing prices as lag features.
      - Creating the target column (next-day Close).
      - Handling missing values.
    """
    # Add technical features using available columns (Adj Close, Close, High, Low, Open, Volume)
    df_symbol = add_technical_features(df_symbol)
    
    # Add rolling features for High, Low, and Volume (7-day window)
    df_rolling = add_rolling_features(df_symbol, window=7)
    # Merge the new rolling features back into the main DataFrame.
    # We reset the index to ensure the merge is aligned on row order.
    df_symbol = pd.concat([df_symbol.reset_index(drop=True), df_rolling.reset_index(drop=True)], axis=1)
    
    # Create lagged features for past LOOKBACK_DAYS closes (using Close)
    for i in range(1, lookback_days + 1):
        df_symbol[f"Close_Lag_{i}"] = df_symbol["Close"].shift(i)

    # Target variable: Next-day Close
    df_symbol["Target"] = df_symbol["Close"].shift(-1)

    # Drop rows with NaN values (caused by rolling calculations and shifting)
    df_symbol.dropna(inplace=True)

    # Ensure DataFrame is sorted by Date
    df_symbol.sort_values(by="Date", inplace=True)

    return df_symbol

#####################################################################
# 6. TIME-BASED TRAIN-VALIDATION SPLIT
#####################################################################
def time_based_split(df, date_col="Date", split_ratio=TRAIN_SPLIT_RATIO):
    """
    Splits df into train/val based on chronological order.
    """
    df = df.sort_values(by=date_col)
    unique_dates = df[date_col].unique()
    cutoff_index = int(len(unique_dates) * split_ratio)
    cutoff_date = unique_dates[cutoff_index]

    df_train = df[df[date_col] <= cutoff_date]
    df_val = df[df[date_col] > cutoff_date]

    return df_train, df_val, cutoff_date

#####################################################################
# 7. TRAIN SVR WITH FEATURE SCALING AND HYPERPARAMETER TUNING
#####################################################################
def train_svr(df_train, lookback_days=LOOKBACK_DAYS):
    """
    Trains an SVR using a combination of lag features and technical/rolling indicators.
    In this example, we use:
      - Lag features for the Close price: Close_Lag_1, ..., Close_Lag_{LOOKBACK_DAYS}
      - Additional technical indicators: SMA_10, SMA_50, EMA_10, Bollinger_Width, MACD, RSI_14, OBV, ATR_14
      - Rolling features: High_7d, Low_7d, Volume_7d
    """
    # Define lag features based on Close prices
    lag_features = [f"Close_Lag_{i}" for i in range(1, lookback_days + 1)]
    
    # Define additional technical indicators
    tech_features = ["SMA_10", "SMA_50", "EMA_10", "Bollinger_Width", "MACD", "RSI_14", "OBV", "ATR_14"]
    
    # Define new rolling features computed above
    rolling_features = ["High_7d", "Low_7d", "Volume_7d"]
    
    # Combine all features
    feature_cols = lag_features + rolling_features
    
    X_train = df_train[feature_cols].values
    y_train = df_train["Target"].values

    # Create a pipeline that scales features and then trains an SVR
    pipeline = make_pipeline(StandardScaler(), SVR(kernel="rbf"))

    # Define grid of hyperparameters for SVR
    param_grid = {
        'svr__C': [0.1, 0.5, 1.0],
        'svr__epsilon': [0.01, 0.1, 0.5],
        'svr__gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"  [INFO] Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

#####################################################################
# 8. PLOT TRAIN & VALIDATION PREDICTIONS
#####################################################################
def plot_train_val(df_train, df_val, y_pred_train, y_pred_val, cutoff_date, symbol):
    """
    Combines train & validation data and plots Real vs. Predicted next-day Close.
    A vertical line indicates the train/validation split.
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
    output_dir = "output_svr"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{symbol}_train_val.png")
    plt.savefig(out_file)
    plt.close()
    print(f"  [INFO] Plot saved to: {out_file}")

#####################################################################
# 9. MAIN FUNCTION
#####################################################################
def main():
    # 1. Load the dataset
    df = load_full_data()

    # 2. Process each target symbol
    for symbol in TARGET_SYMBOLS:
        print(f"\n[INFO] Processing symbol: {symbol}")
        df_symbol = filter_by_symbol(df, symbol)
        if df_symbol.empty:
            continue

        # 3. Preprocess data: add technical features, rolling features, lag features, and target
        df_svr = preprocess_data_svr(df_symbol)

        # Check if we have sufficient data
        if len(df_svr) < 50:
            print(f"[WARNING] {symbol}: insufficient data after preprocessing.")
            continue

        # 4. Time-based train-validation split
        df_train, df_val, cutoff_date = time_based_split(df_svr, "Date", TRAIN_SPLIT_RATIO)
        print(f"  [INFO] {symbol} -> Train size: {len(df_train)}, Validation size: {len(df_val)}")

        # 5. Train the SVR model with hyperparameter tuning
        model = train_svr(df_train)

        # 6. Define the same feature columns for prediction
        feature_cols = (
            [f"Close_Lag_{i}" for i in range(1, LOOKBACK_DAYS + 1)] +
            #["SMA_10", "SMA_50", "EMA_10", "Bollinger_Width", "MACD", "RSI_14", "OBV", "ATR_14"] +
            ["High_7d", "Low_7d", "Volume_7d"]
        )

        # 7. Make predictions on both training and validation sets
        y_pred_train = model.predict(df_train[feature_cols].values)
        y_pred_val = model.predict(df_val[feature_cols].values)

        # 8. Calculate and print Mean Squared Error for train and validation sets
        train_mse = mean_squared_error(df_train["Target"], y_pred_train)
        val_mse = mean_squared_error(df_val["Target"], y_pred_val)
        print(f"  [TRAIN] MSE: {train_mse:.4f}")
        print(f"  [VAL]   MSE: {val_mse:.4f}")

        # 9. Plot the results
        plot_train_val(df_train, df_val, y_pred_train, y_pred_val, cutoff_date, symbol)

if __name__ == "__main__":
    main()
