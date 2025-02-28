import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error

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


def load_full_data():
    csv_path = os.path.join("sp500_stocks", "sp500_stocks.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"[ERROR] CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def filter_by_symbol(df, symbol):
    df_symbol = df[df['Symbol'] == symbol].copy()
    return df_symbol


def preprocess_data_prophet(df_symbol):
    df_symbol = df_symbol.dropna(subset=['Close'])
    df_symbol.fillna(method='bfill', inplace=True)
    df_symbol.fillna(method='ffill', inplace=True)

    if 'Date' in df_symbol.columns:
        df_symbol['Date'] = pd.to_datetime(df_symbol['Date'])
    df_symbol.sort_values(by='Date', inplace=True)

    df_prophet = df_symbol[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    return df_prophet


def split_data(df_prophet):
    train_size = int(len(df_prophet) * 0.8)
    train = df_prophet.iloc[:train_size]
    val = df_prophet.iloc[train_size:]
    return train, val


def prophet_forecast(train, val):
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(train)

    future = model.make_future_dataframe(periods=len(val), freq='D')
    forecast = model.predict(future)

    train_rmse = np.sqrt(mean_squared_error(train['y'], forecast['yhat'][:len(train)]))
    val_rmse = np.sqrt(mean_squared_error(val['y'], forecast['yhat'][len(train):]))

    return forecast, model, train_rmse, val_rmse


def main():
    df = load_full_data()

    for symbol in TARGET_SYMBOLS:
        print(f"\n[INFO] Processing symbol: {symbol}")
        df_symbol = filter_by_symbol(df, symbol)
        if df_symbol.empty:
            continue

        df_prophet = preprocess_data_prophet(df_symbol)
        if len(df_prophet) < 50:
            print(f"[WARNING] {symbol}: insufficient data after preprocessing.")
            continue

        train, val = split_data(df_prophet)
        forecast, model, train_rmse, val_rmse = prophet_forecast(train, val)

        print(f"[INFO] {symbol} - RMSE Train: {train_rmse:.2f}, RMSE Validation: {val_rmse:.2f}")

        fig1 = model.plot(forecast, xlabel='Date', ylabel='Price (Close)')
        fig1.suptitle(f"2-Year Forecast for {symbol}", fontsize=14)
        fig1.savefig(f"{symbol}_forecast.png")
        plt.close(fig1)

        fig2 = model.plot_components(forecast)
        fig2.suptitle(f"Forecast Components for {symbol}", fontsize=14)
        fig2.savefig(f"{symbol}_forecast_components.png")
        plt.close(fig2)


if __name__ == "__main__":
    main()