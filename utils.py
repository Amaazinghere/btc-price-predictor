import pandas as pd
import numpy as np
from binance.client import Client
import joblib
import os

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

def fetch_binance_data(symbol='BTCUSDT', interval='1d', lookback_days=400):
    klines = client.get_historical_klines(symbol, interval, f"{lookback_days} days ago UTC")
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_asset_volume', 'Number_of_trades',
        'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
    ])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    return df[['Date', 'Close', 'Volume']]

def prepare_features(df):
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_MA20'] + (2 * df['BB_std'])
    df['BB_lower'] = df['BB_MA20'] - (2 * df['BB_std'])
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Volume_Change_1d'] = df['Volume'].pct_change(1)
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['Close_t-1'] = df['Close'].shift(1)
    df['Close_t-2'] = df['Close'].shift(2)
    df['MA_5_t-1'] = df['MA_5'].shift(1)
    df['MA_10_t-1'] = df['MA_10'].shift(1)
    df['RSI_14_t-1'] = df['RSI_14'].shift(1)
    df['EMA_10_t-1'] = df['EMA_10'].shift(1)
    df['MACD_t-1'] = df['MACD'].shift(1)
    df = df.dropna()
    return df

def predict_price(df, model, f_scaler, t_scaler):
    feature_cols = [
        'Close_t-1', 'Close_t-2', 'MA_5_t-1', 'MA_10_t-1', 'EMA_5', 'EMA_10', 'EMA_10_t-1',
        'RSI_14_t-1', 'MACD', 'MACD_t-1', 'MACD_signal',
        'BB_upper', 'BB_lower',
        'Log_Return', 'Return_5d', 'Return_10d',
        'Volume', 'Volume_Change_1d', 'Volume_MA_5'
    ]
    X = df[feature_cols]
    X_scaled = f_scaler.transform(X)
    y_pred_scaled = model.predict(X_scaled)
    y_pred_delta = t_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    latest_price = df['Close'].values
    predicted_price = latest_price + y_pred_delta
    return df['Date'].values, predicted_price