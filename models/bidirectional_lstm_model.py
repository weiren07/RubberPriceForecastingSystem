import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
print("Loading dataset...")
data = pd.read_csv('data/preprocessed_data.csv', parse_dates=['date'], index_col='date')
print("Dataset loaded successfully.")

# Apply log transformation to specific column 'SMR20'
print("Applying log transformation to 'SMR20' column...")
data['SMR20'] = np.log(data['SMR20'])
print("Log transformation applied.")

# Scale the data
print("Scaling data...")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print("Data scaling completed.")

# Define the number of time steps and features
n_steps = 3
n_features = scaled_data.shape[1]

# Define the input and output data
print("Preparing input and output data...")
X, y = [], []
for i in range(n_steps, len(scaled_data)):
    X.append(scaled_data[i-n_steps:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)
print("Input and output data prepared.")

# Define the sizes of the training, validation, and testing datasets
train_size = int(0.6 * len(X))
val_size = int(0.2 * len(X))
test_size = len(X) - train_size - val_size

# Split the data into training, validation, and testing sets
print("Splitting data into training, validation, and testing sets...")
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
print("Data split completed.")

# Define the model
print("Building the model...")
model = Sequential()
model.add(Bidirectional(LSTM(units=70, activation='relu', input_shape=(n_steps, n_features), return_sequences=True)))
model.add(Bidirectional(LSTM(units=16, activation='relu', return_sequences=True)))
model.add(Bidirectional(LSTM(units=8, activation='relu')))
model.add(Dense(units=n_features))
print("Model architecture defined.")

# Compile the model
print("Compiling the model...")
model.compile(optimizer='adam', loss='mse')
print("Model compiled.")

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
print("Model training completed.")

# Evaluate the model
print("Evaluating the model on validation and test sets...")
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Inverse transform the predictions and actual values
print("Applying inverse transformations...")
y_pred_val = scaler.inverse_transform(y_pred_val)
y_val = scaler.inverse_transform(y_val)
y_pred_test = scaler.inverse_transform(y_pred_test)
y_test = scaler.inverse_transform(y_test)

# Transform 'SMR20' back to the original scale
y_pred_val[:, data.columns.get_loc('SMR20')] = np.exp(y_pred_val[:, data.columns.get_loc('SMR20')])
y_val[:, data.columns.get_loc('SMR20')] = np.exp(y_val[:, data.columns.get_loc('SMR20')])
y_pred_test[:, data.columns.get_loc('SMR20')] = np.exp(y_pred_test[:, data.columns.get_loc('SMR20')])
y_test[:, data.columns.get_loc('SMR20')] = np.exp(y_test[:, data.columns.get_loc('SMR20')])
print("Inverse transformation applied.")

# Calculate evaluation metrics
print("Calculating evaluation metrics...")
mae_val = mean_absolute_error(y_val, y_pred_val)
mse_val = mean_squared_error(y_val, y_pred_val)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_pred_val)
mape_val = np.mean(np.abs((y_val - y_pred_val) / y_val)) * 100

mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)
mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

# Print the evaluation metrics
print("Validation Set Metrics:")
print("MAE:", mae_val)
print("MSE:", mse_val)
print("RMSE:", rmse_val)
print("R^2:", r2_val)
print("MAPE:", mape_val)

print("Testing Set Metrics:")
print("MAE:", mae_test)
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("R^2:", r2_test)
print("MAPE:", mape_test)

print("Script finished successfully.")
