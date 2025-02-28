#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#####################################################################
# User Configuration
#####################################################################
# List of stock symbols to process
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

# Path to the CSV file containing stock data
STOCK_CSV_PATH = os.path.join("sp500_stocks", "sp500_stocks.csv")

# Number of past days to use for lag features
LOOKBACK_DAYS = 5

# Train/test split ratio (0.9 means 90% training data)
TRAIN_SPLIT_RATIO = 0.8

#####################################################################
# 1. Data Loading and Initial Processing
#####################################################################
def load_full_data():
    """
    Load and perform initial processing of stock data from CSV file.
    
    Returns:
        pandas.DataFrame: Processed stock data with handled NaN values
    """
    if not os.path.exists(STOCK_CSV_PATH):
        sys.exit(f"[ERROR] CSV file not found at: {STOCK_CSV_PATH}")

    print(f"[INFO] Loading dataset from: {STOCK_CSV_PATH}")
    df = pd.read_csv(STOCK_CSV_PATH)

    # Convert Date column to datetime
    if 'Date' in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # Sort data by Symbol and Date
    df.sort_values(by=["Symbol", "Date"], inplace=True)

    # Handle missing values in Close price using forward and backward fill
    df["Close"] = df.groupby("Symbol")["Close"].ffill().bfill()

    print(f"[INFO] DataFrame shape: {df.shape}")
    print(df.head(3))
    return df

#####################################################################
# 2. Symbol Filtering
#####################################################################
def filter_by_symbol(df, symbol):
    """
    Extract data for a specific stock symbol.
    
    Args:
        df (pandas.DataFrame): Full stock dataset
        symbol (str): Stock symbol to filter
        
    Returns:
        pandas.DataFrame: Data for the specified symbol
    """
    df_symbol = df[df['Symbol'] == symbol].copy()
    if df_symbol.empty:
        print(f"[WARNING] No data found for {symbol}.")
    else:
        print(f"[INFO] {symbol}: {len(df_symbol)} rows found.")
    return df_symbol

#####################################################################
# 3. Technical Indicators
#####################################################################
def add_technical_features(df):
    """
    Add technical analysis indicators to the dataset.
    
    Features added:
    - Moving averages (SMA, EMA)
    - Bollinger Bands
    - MACD
    - RSI
    - ATR
    - OBV
    
    Args:
        df (pandas.DataFrame): Stock data
        
    Returns:
        pandas.DataFrame: Data with added technical indicators
    """
    df = df.copy()

    # Simple Moving Averages
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # Bollinger Bands (20-day)
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["STD_20"] = df["Close"].rolling(window=20).std()
    df["Bollinger_Upper"] = df["SMA_20"] + 2 * df["STD_20"]
    df["Bollinger_Lower"] = df["SMA_20"] - 2 * df["STD_20"]
    df["Bollinger_Width"] = (df["Bollinger_Upper"] - df["Bollinger_Lower"]) / df["SMA_20"]

    # MACD (Moving Average Convergence Divergence)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Average True Range (ATR)
    df["Prev_Close"] = df["Close"].shift(1)
    df["High_Low"] = df["High"] - df["Low"]
    df["High_PrevClose"] = np.abs(df["High"] - df["Prev_Close"])
    df["Low_PrevClose"] = np.abs(df["Low"] - df["Prev_Close"])
    df["True_Range"] = df[["High_Low", "High_PrevClose", "Low_PrevClose"]].max(axis=1)
    df["ATR_14"] = df["True_Range"].rolling(window=14).mean()

    # Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # On-Balance Volume (OBV)
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # Clean up intermediate columns
    df.drop(columns=["Prev_Close", "High_Low", "High_PrevClose", "Low_PrevClose"], inplace=True)

    return df

#####################################################################
# 4. Rolling Features
#####################################################################
def add_rolling_features(df, window=7):
    """
    Add rolling mean features for High, Low, and Volume.
    
    Args:
        df (pandas.DataFrame): Stock data
        window (int): Rolling window size in days
        
    Returns:
        pandas.DataFrame: Rolling mean features
    """
    lag_features = ["High", "Low", "Volume"]
    df_rolled = df[lag_features].rolling(window=window, min_periods=0)
    df_mean = df_rolled.mean().shift(1)
    df_mean = df_mean.astype(np.float32)
    df_mean.columns = [f"{col}_{window}d" for col in lag_features]
    return df_mean

#####################################################################
# 5. Data Preprocessing
#####################################################################
def preprocess_data_xgb(df_symbol, lookback_days=LOOKBACK_DAYS):
    """
    Prepare data for XGBoost training by adding all features and handling missing values.
    
    Args:
        df_symbol (pandas.DataFrame): Single symbol stock data
        lookback_days (int): Number of past days to use for lag features
        
    Returns:
        pandas.DataFrame: Processed data ready for model training
    """
    # Add technical indicators
    df_symbol = add_technical_features(df_symbol)
    
    # Add rolling features
    df_rolling = add_rolling_features(df_symbol, window=7)
    df_symbol = pd.concat([df_symbol.reset_index(drop=True), 
                          df_rolling.reset_index(drop=True)], axis=1)
    
    # Create lag features
    for i in range(1, lookback_days + 1):
        df_symbol[f"Close_Lag_{i}"] = df_symbol["Close"].shift(i)

    # Create target variable (next day's close price)
    df_symbol["Target"] = df_symbol["Close"].shift(-1)

    # Remove rows with missing values
    df_symbol.dropna(inplace=True)

    # Ensure chronological order
    df_symbol.sort_values(by="Date", inplace=True)

    return df_symbol

#####################################################################
# 6. Train-Validation Split
#####################################################################
def time_based_split(df, date_col="Date", split_ratio=TRAIN_SPLIT_RATIO):
    """
    Split data into training and validation sets based on time.
    
    Args:
        df (pandas.DataFrame): Data to split
        date_col (str): Name of date column
        split_ratio (float): Ratio for train/validation split
        
    Returns:
        tuple: (training_data, validation_data, cutoff_date)
    """
    df = df.sort_values(by=date_col)
    unique_dates = df[date_col].unique()
    cutoff_index = int(len(unique_dates) * split_ratio)
    cutoff_date = unique_dates[cutoff_index]

    df_train = df[df[date_col] <= cutoff_date]
    df_val = df[df[date_col] > cutoff_date]

    return df_train, df_val, cutoff_date


def save_performance_metrics(symbol, train_mse, val_mse, model_type="xgb_advanced"):
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
# 7. Model Training
#####################################################################
def train_xgb(df_train, lookback_days=LOOKBACK_DAYS):
    """
    Train XGBoost model with hyperparameter tuning.
    
    Args:
        df_train (pandas.DataFrame): Training data
        lookback_days (int): Number of past days used in features
        
    Returns:
        XGBRegressor: Trained model with best parameters
    """
    # Define feature sets
    lag_features = [f"Close_Lag_{i}" for i in range(1, lookback_days + 1)]
    tech_features = ["SMA_10", "SMA_50", "EMA_10", "Bollinger_Width", 
                    "MACD", "RSI_14", "OBV", "ATR_14"]
    rolling_features = ["High_7d", "Low_7d", "Volume_7d"]
    
    feature_cols = lag_features + rolling_features
    
    X_train = df_train[feature_cols].values
    y_train = df_train["Target"].values

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }

    # Create base model
    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )

    # Perform grid search
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"  [INFO] Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

#####################################################################
# 8. Visualization
#####################################################################
def plot_train_val(df_train, df_val, y_pred_train, y_pred_val, cutoff_date, symbol):
    """
    Create and save visualization of model predictions.
    
    Args:
        df_train (pandas.DataFrame): Training data
        df_val (pandas.DataFrame): Validation data
        y_pred_train (array): Training predictions
        y_pred_val (array): Validation predictions
        cutoff_date (datetime): Train/validation split date
        symbol (str): Stock symbol
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

    output_dir = "output_xgb_advanced"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{symbol}_train_val.png")
    plt.savefig(out_file)
    plt.close()
    print(f"  [INFO] Plot saved to: {out_file}")

def analyze_model_performance():
    """
    Analyze and compare performance metrics across different models.
    Creates summary visualizations and statistics.
    """
    metrics_file = os.path.join("performance_metrics", "model_performance.csv")
    if not os.path.exists(metrics_file):
        print("[ERROR] No performance metrics file found")
        return
        
    # Read performance data
    df = pd.read_csv(metrics_file)
    
    # Calculate average metrics by model type
    summary = df.groupby('Model').agg({
        'Train_MSE': ['mean', 'std'],
        'Val_MSE': ['mean', 'std'],
        'Train_RMSE': ['mean', 'std'],
        'Val_RMSE': ['mean', 'std']
    }).round(4)
    
    # Save summary statistics
    summary.to_csv(os.path.join("performance_metrics", "model_comparison_summary.csv"))
    
    # Create box plots for validation RMSE by model
    plt.figure(figsize=(10, 6))
    plt.title("Model Performance Comparison - Validation RMSE")
    plt.boxplot([df[df['Model'] == model]['Val_RMSE'] for model in df['Model'].unique()],
                labels=df['Model'].unique())
    plt.ylabel('Validation RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join("performance_metrics", "model_comparison_boxplot.png"))
    plt.close()
    
    print("\n[INFO] Model Performance Summary:")
    print(summary)
    print("\n[INFO] Performance analysis saved to 'performance_metrics' directory")

#####################################################################
# 9. Main Function
#####################################################################
def main():
    """
    Main execution function that coordinates the entire process.
    """
    # Load dataset
    df = load_full_data()

    # Process each symbol
    for symbol in TARGET_SYMBOLS:
        print(f"\n[INFO] Processing symbol: {symbol}")
        
        # Filter data for current symbol
        df_symbol = filter_by_symbol(df, symbol)
        if df_symbol.empty:
            continue

        # Preprocess data
        df_xgb = preprocess_data_xgb(df_symbol)

        # Check data sufficiency
        if len(df_xgb) < 50:
            print(f"[WARNING] {symbol}: insufficient data after preprocessing.")
            continue

        # Split data
        df_train, df_val, cutoff_date = time_based_split(df_xgb, "Date", TRAIN_SPLIT_RATIO)
        print(f"  [INFO] {symbol} -> Train size: {len(df_train)}, Validation size: {len(df_val)}")

        # Train model
        model = train_xgb(df_train)

        # Define features for prediction
        feature_cols = (
            [f"Close_Lag_{i}" for i in range(1, LOOKBACK_DAYS + 1)] +
            ["High_7d", "Low_7d", "Volume_7d"]
        )

        # Make predictions
        y_pred_train = model.predict(df_train[feature_cols].values)
        y_pred_val = model.predict(df_val[feature_cols].values)

        # Calculate error metrics
        # Calculate error metrics
        train_mse = mean_squared_error(df_train["Target"], y_pred_train)
        val_mse = mean_squared_error(df_val["Target"], y_pred_val)
        print(f"  [TRAIN] MSE: {train_mse:.4f}")
        print(f"  [VAL]   MSE: {val_mse:.4f}")

        # Save performance metrics
        save_performance_metrics(symbol, train_mse, val_mse)

        # Create visualization
        plot_train_val(df_train, df_val, y_pred_train, y_pred_val, cutoff_date, symbol)

if __name__ == "__main__":
    main()
    analyze_model_performance()