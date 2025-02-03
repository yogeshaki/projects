# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Load data from a CSV file
data = pd.read_csv(r'C:\Users\yogesh\Downloads\yostock4.csv')

# Step 2: Check if 'Unnamed: 0' is the date column
print(data.columns)

# If 'Unnamed: 0' is actually the date, rename it and convert it to datetime
data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])

# Step 3: Data Preprocessing
data = data.dropna()  # Drop missing values

# Assuming your CSV has a 'Close' column representing the closing price
data['MA_20'] = data['Close'].rolling(window=20).mean()  # Create 20-day moving average
data['Target'] = data['Close'].shift(-1)  # Target is the next day's closing price
data = data.dropna()  # Drop rows with NaN values after shift

# Step 4: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.plot(data['Date'], data['MA_20'], label='20-Day MA', color='orange')
plt.title('Stock Closing Price with 20-Day Moving Average')
plt.legend()
plt.show()

# Step 5: Prepare the features (X) and target (y)
X = data[['MA_20']]
y = data['Target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Evaluate Linear Regression model
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Linear Regression RMSE: {rmse_lr}')

# Step 7: LSTM Model
# Reshape data for LSTM (3D shape: samples, timesteps, features)
X_train_lstm = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=False, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# Train the LSTM model with validation data
history = lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))

# Make predictions with LSTM
y_pred_lstm = lstm_model.predict(X_test_lstm)

# Evaluate LSTM model
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
print(f'LSTM RMSE: {rmse_lstm}')

# Step 8: Visualization of Actual vs Predicted Prices
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Price', color='blue')
plt.plot(y_pred, label='Predicted Price (LR)', color='red', linestyle='dashed')
plt.plot(y_pred_lstm, label='Predicted Price (LSTM)', color='green', linestyle='dotted')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()
