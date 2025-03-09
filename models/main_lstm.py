#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#####################################################################
# User Configuration
#####################################################################
TARGET_SYMBOLS = [
    "MSFT", "AMZN", "TSLA", "GOOG", "META",
    "RJF", "V", 
    "NFLX", "DIS", "FOX"
]

STOCK_CSV_PATH = os.path.join("sp500_stocks", "sp500_stocks.csv")

# Number of past days to include in LSTM sequence
LOOKBACK_DAYS = 60  

# Ratio of training data vs. total
TRAIN_SPLIT_RATIO = 0.8

# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 32

#####################################################################
# 1. LOAD DATA
#####################################################################
def load_full_data():
    """
    Loads the CSV file containing all S&P 500 stock data.
    Expects columns including Date, Symbol, Close.
    """
    if not os.path.exists(STOCK_CSV_PATH):
        sys.exit(f"[ERROR] CSV file not found at: {STOCK_CSV_PATH}")

    print(f"[INFO] Loading dataset from: {STOCK_CSV_PATH}")
    df = pd.read_csv(STOCK_CSV_PATH)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by=["Symbol", "Date"], inplace=True)

    # Forward and backward fill missing Close prices
    df["Close"] = df.groupby("Symbol")["Close"].ffill().bfill()

    return df

#####################################################################
# 2. FILTER BY SYMBOL
#####################################################################
def filter_by_symbol(df, symbol):
    """Returns only the rows for the specified stock symbol."""
    df_symbol = df[df['Symbol'] == symbol].copy()
    if df_symbol.empty:
        print(f"[WARNING] No data found for {symbol}.")
    return df_symbol

#####################################################################
# 3. BUILD LSTM MODEL
#####################################################################
def build_lstm_model(input_shape):
    """Defines the LSTM model architecture."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model

#####################################################################
# 4. CREATE TRAIN/TEST SPLIT & PREVENT LEAKAGE
#####################################################################
def create_train_test_data(df_symbol, lookback_days=60, train_ratio=0.8):
    """
    Steps:
      1) Compute train/test split index (chronological).
      2) Fit scaler on *training portion only* to avoid leakage.
      3) Transform entire series with that fitted scaler.
      4) Build lookback sequences, assigning them to train or test
         based on their last index.
    """
    # Ensure data is sorted by date
    df_symbol = df_symbol.sort_values("Date").reset_index(drop=True)
    prices = df_symbol["Close"].values
    n = len(prices)
    if n <= lookback_days:
        print("Not enough data for lookback sequences.")
        return None, None, None, None, None, None, None

    # 1) Train/test split index
    split_idx = int(n * train_ratio)

    # 2) Fit MinMaxScaler ONLY on the training portion
    train_prices = prices[:split_idx].reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_prices)

    # 3) Transform the entire dataset with the *train-fitted* scaler
    scaled_prices = scaler.transform(prices.reshape(-1, 1))

    # 4) Build sequences
    X_train, y_train = [], []
    X_test, y_test = [], []

    for i in range(lookback_days, n):
        X_seq = scaled_prices[i - lookback_days:i, 0]
        y_val = scaled_prices[i, 0]

        # Determine if this index is in the training set or test set
        # We'll say if 'i' < split_idx, then it's part of training sequences
        # (so the label is i, which is in the training region).
        if i < split_idx:
            X_train.append(X_seq)
            y_train.append(y_val)
        else:
            X_test.append(X_seq)
            y_test.append(y_val)

    # Convert to numpy
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Reshape for LSTM (samples, timesteps, features=1)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, scaler, split_idx, df_symbol

#####################################################################
# 5. TRAIN MODEL
#####################################################################
def train_lstm(model, X_train, y_train, epochs=50, batch_size=32):
    """Trains the LSTM model."""
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

#####################################################################
# 6. MAKE PREDICTIONS
#####################################################################
def make_predictions(model, X, scaler):
    """Predicts stock prices and inverts scaling."""
    pred_scaled = model.predict(X)
    return scaler.inverse_transform(pred_scaled)

#####################################################################
# 7. RMSE CALC
#####################################################################
def compute_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

#####################################################################
# 8. COMBINED TRAIN/VAL PLOT WITH RED LINE
#####################################################################
def plot_train_val_predictions(df_symbol,
                               split_idx,
                               lookback_days,
                               y_train_pred,
                               y_test_pred,
                               scaler,
                               symbol):
    """
    Plots the real vs. predicted close for both the training and test sets
    in one figure, with a red vertical line at the train/val boundary.

    df_symbol:  Full symbol DataFrame (chronologically sorted)
    split_idx:  Index in df_symbol where train->test splits
    lookback_days: how many days used for each LSTM sequence
    y_train_pred: predictions (in original scale) for training samples
    y_test_pred: predictions (in original scale) for test samples
    scaler: fitted MinMaxScaler (for inverse transform if needed)
    symbol:  string for plot title
    """
    # We'll reconstruct the "true" values for the last day of each sequence
    prices = df_symbol["Close"].values
    dates = df_symbol["Date"].values

    # For the first (lookback_days) samples, we have no predictions
    # so the training predictions start at index = lookback_days
    # and go up to split_idx-1
    train_pred_indices = np.arange(lookback_days, split_idx)
    test_pred_indices  = np.arange(split_idx, len(prices))

    # Note: Because we took the label from the i-th position,
    # the shape of y_train_pred is (split_idx - lookback_days).
    # Similarly for test: shape is (len(prices) - split_idx).

    # Build arrays for plotting
    real_prices_train = prices[train_pred_indices]  # actual train region
    real_prices_test  = prices[test_pred_indices]   # actual test region

    # Convert everything to numpy arrays for plotting
    train_dates = dates[train_pred_indices]
    test_dates  = dates[test_pred_indices]

    plt.figure(figsize=(10, 6))
    plt.title(f"{symbol} - LSTM Train & Validation Predictions")

    # Plot real prices
    plt.plot(train_dates, real_prices_train, label="Train Real", linestyle="-")
    plt.plot(test_dates,  real_prices_test,  label="Validation Real", linestyle="-")

    # Plot predictions
    plt.plot(train_dates, y_train_pred, label="Train Predicted", linestyle="--")
    plt.plot(test_dates,  y_test_pred,  label="Validation Predicted", linestyle="--")

    # Mark the train/val split date with a vertical line
    cutoff_date = dates[split_idx]
    plt.axvline(x=cutoff_date, color="red", linestyle="--", label="Train/Val Split")

    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_dir = "output_lstm"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{symbol}_train_val.png")
    plt.savefig(out_file)
    plt.close()

    print(f"[INFO] {symbol}: Plot saved to {out_file}")

#####################################################################
# MAIN FUNCTION
#####################################################################
def main():
    df = load_full_data()

    # Remove old performance file if you want a fresh start each run
    output_file = "performance_metrics_lstm.csv"
    if os.path.exists(output_file):
        os.remove(output_file)

    for symbol in TARGET_SYMBOLS:
        print(f"\n[INFO] Processing {symbol}...")

        # 1) Filter data by symbol
        df_symbol = filter_by_symbol(df, symbol)
        if len(df_symbol) < LOOKBACK_DAYS:
            print(f"  [WARNING] Not enough data points for {symbol} (need > {LOOKBACK_DAYS}). Skipping.")
            continue

        # 2) Create train/test data WITHOUT leakage
        results = create_train_test_data(df_symbol, lookback_days=LOOKBACK_DAYS, 
                                         train_ratio=TRAIN_SPLIT_RATIO)
        if results[0] is None:
            # Means not enough data to form sequences
            continue
        X_train, y_train, X_test, y_test, scaler, split_idx, df_symbol = results

        # Inverse-transform the actual y for error metrics
        y_train_real = scaler.inverse_transform(y_train.reshape(-1,1))
        y_test_real  = scaler.inverse_transform(y_test.reshape(-1,1))

        # 3) Build model
        model = build_lstm_model((X_train.shape[1], 1))

        # 4) Train the LSTM
        model = train_lstm(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

        # 5) Make predictions on train and test sets
        y_train_pred = make_predictions(model, X_train, scaler)
        y_test_pred  = make_predictions(model, X_test,  scaler)

        # 6) Compute RMSE
        train_rmse = compute_rmse(y_train_real, y_train_pred)
        test_rmse = compute_rmse(y_test_real, y_test_pred)
        print(f"  [TRAIN] RMSE: {train_rmse:.4f}")
        print(f"  [VAL]   RMSE: {test_rmse:.4f}")

        # 7) Save Performance Metrics
        metrics_df = pd.DataFrame({
            'Symbol': [symbol],
            'Train_RMSE': [train_rmse],
            'Val_RMSE': [test_rmse]
        })
        metrics_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

        # 8) Plot the combined train/val with vertical split
        plot_train_val_predictions(df_symbol,
                                   split_idx,
                                   LOOKBACK_DAYS,
                                   y_train_pred.flatten(),
                                   y_test_pred.flatten(),
                                   scaler,
                                   symbol)

    print("\n[INFO] LSTM Model Training and Prediction Completed.")

if __name__ == "__main__":
    main()
