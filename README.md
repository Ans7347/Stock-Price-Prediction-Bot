This Python trading bot is designed to predict stock prices using an LSTM (Long Short-Term Memory) neural network. Here's a summary of what it does:

Data Collection:
Historical Data: The bot downloads historical stock data from Alpha Vantage for a specified ticker (e.g., 'AAPL') between two dates.
Live Data: It fetches live intraday stock data at one-minute intervals.

Data Preprocessing:
Scaling: The data is scaled between 0 and 1 using MinMaxScaler to normalize the features.
LSTM Preparation: The preprocessed data is prepared for the LSTM model by creating sequences of time_step (default 60s) with corresponding labels (target 'Close' price).

Model Creation:
The bot creates an LSTM model with two LSTM layers, dropout layers to prevent overfitting, and dense layers for prediction.

Training:
The model is trained on the historical data with early stopping and model checkpoint callbacks to optimize performance.

Real-Time Prediction:
The bot continuously updates the model with new live data and generates predictions every minute.

Predictions are saved to a file called predictions.txt.
The bot is designed for continuous, real-time operation, constantly updating its predictions based on the latest data.

Note: Run the cpp file first and then the python file.
