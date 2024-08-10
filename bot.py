import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# Set up Alpha Vantage API
API_KEY = 'ENTER_YOUR_API_KEY'
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Download historical stock data
def download_data(ticker, start_date, end_date):
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    data = data.loc[start_date:end_date]
    return data

# Preprocess the data
def preprocess_data(data):
    data = data[['1. open', '2. high', '3. low', '4. close', '5. volume']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Prepare data for LSTM
def prepare_lstm_data(data, time_step=60):
    X, Y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])
        Y.append(data[i, 3])  # Assuming 'Close' price is the target
    X, Y = np.array(X), np.array(Y)
    return X, Y

# Create LSTM model
def create_model(input_shape):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(units=100, return_sequences=False),
        Dropout(0.3),
        Dense(units=50, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Fetch live data
def fetch_live_data(ticker):
    live_data, _ = ts.get_intraday(symbol=ticker, interval='1min', outputsize='compact')
    return live_data

# Preprocess live data
def preprocess_live_data(live_data, scaler):
    live_data = live_data[['1. open', '2. high', '3. low', '4. close', '5. volume']]
    scaled_data = scaler.transform(live_data)
    return scaled_data

# Update model with new data
def update_model_with_new_data(model, new_data, scaler, time_step=60):
    scaled_data = preprocess_live_data(new_data, scaler)
    X, Y = prepare_lstm_data(scaled_data, time_step)
    model.fit(X, Y, epochs=5, batch_size=64, verbose=1, callbacks=[EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
                         ModelCheckpoint('best_model.keras', save_best_only=True)])

# Predict future prices
def predict_with_updated_model(model, new_data, scaler):
    scaled_data = preprocess_live_data(new_data, scaler)
    X, _ = prepare_lstm_data(scaled_data)
    return model.predict(X)

# Save model predictions to a file
def save_predictions(predictions, filename="predictions.txt"):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred[0]}\n")

# Main loop for real-time training and prediction
def main():
    # Initial data download and model training
    data = download_data('AAPL', '2022-01-01', '2024-07-07')
    scaled_data, scaler = preprocess_data(data)
    X, Y = prepare_lstm_data(scaled_data)
    model = create_model((X.shape[1], X.shape[2]))
    
    # Model training with early stopping and checkpoint
    model.fit(X, Y, epochs=20, batch_size=64, callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                         ModelCheckpoint('best_model.keras', save_best_only=True)])
    
    while True:
        live_data = fetch_live_data('AAPL')
        update_model_with_new_data(model, live_data, scaler)
        predictions = predict_with_updated_model(model, live_data, scaler)
        save_predictions(predictions)
        time.sleep(60)  # Wait for a minute before fetching new data

if __name__ == "__main__":
    main()
