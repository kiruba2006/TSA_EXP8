# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 28/10/2025


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# Read the Netflix stock dataset
data = pd.read_csv("Netflix_stock_data.csv")

# Ensure proper datetime format and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Focus on the 'Close' column (you can change to 'Adj Close' if needed)
stock_data = data[['Close']]

# Display dataset information
print("Shape of the dataset:", stock_data.shape)
print("First 10 rows of the dataset:")
print(stock_data.head(10))

# Plot Original Dataset
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Original Netflix Closing Price', color='blue')
plt.title('Netflix Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid()
plt.show()

# Moving Average with window sizes 5 and 10
rolling_mean_5 = stock_data['Close'].rolling(window=5).mean()
rolling_mean_10 = stock_data['Close'].rolling(window=10).mean()

# Display first few values
print("\nRolling Mean (window=5):\n", rolling_mean_5.head(10))
print("\nRolling Mean (window=10):\n", rolling_mean_10.head(20))

# Plot Moving Average
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (5 days)', color='orange')
plt.plot(rolling_mean_10, label='Moving Average (10 days)', color='green')
plt.title('Moving Average of Netflix Stock Price')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid()
plt.show()

# Resample data monthly (to make it smoother for forecasting)
data_monthly = stock_data.resample('MS').mean()

# Scale the data
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

# Ensure positive values for multiplicative seasonality
scaled_data = scaled_data + 1

# Split data into train/test
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

# Exponential Smoothing model
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()

# Forecast
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot train, test, predictions
ax = train_data.plot(label='Train Data')
test_predictions_add.plot(ax=ax, label='Predictions')
test_data.plot(ax=ax, label='Test Data')
ax.legend()
ax.set_title('Netflix Stock Forecast (Visual Evaluation)')
plt.show()

# RMSE
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("Root Mean Squared Error (RMSE):", rmse)
print("Standard Deviation:", np.sqrt(scaled_data.var()))
print("Mean:", scaled_data.mean())

# Forecast for one-fourth of total data length
model_full = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model_full.forecast(steps=int(len(scaled_data)/4))

# Plot final predictions
ax = scaled_data.plot(label='Monthly Netflix Stock (Scaled)')
predictions.plot(ax=ax, label='Future Predictions')
ax.set_xlabel('Date')
ax.set_ylabel('Scaled Closing Price')
ax.set_title('Netflix Stock Price Prediction (Next Period)')
ax.legend()
plt.show()
```

### OUTPUT:
Original Dataset

<img width="445" height="322" alt="image" src="https://github.com/user-attachments/assets/6629dc12-3b62-4692-bebe-da4837430eeb" />

<img width="1318" height="682" alt="image" src="https://github.com/user-attachments/assets/f27abdc8-57c4-476d-9cc4-6a052e610302" />


Moving Average

<img width="804" height="790" alt="image" src="https://github.com/user-attachments/assets/425a52e9-28af-408d-ab29-dbadd1153fae" />



Plot Transform Dataset

<img width="1343" height="686" alt="image" src="https://github.com/user-attachments/assets/fcdd5474-7952-4471-b4c3-2daaeebbf0dd" />


Exponential Smoothing

<img width="816" height="584" alt="image" src="https://github.com/user-attachments/assets/4c302020-951e-4c70-8c29-3f02b4b39e20" />

<img width="571" height="82" alt="image" src="https://github.com/user-attachments/assets/9f1ad981-cd5c-4e17-a654-a35a3bf62271" />

<img width="931" height="583" alt="image" src="https://github.com/user-attachments/assets/ccf79e28-1f04-4108-aabf-73723954c23b" />


### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
