
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
data = pd.read_csv('data/modified_data.csv', parse_dates=['date'], index_col='date')

# Apply log transformation to specific column 'SMR20'
data['SMR20'] = np.log(data['SMR20'])

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Define the number of time steps and features
n_steps = 3
n_features = scaled_data.shape[1]

# Define the input and output data
X, y = [], []
for i in range(n_steps, len(scaled_data)):
    X.append(scaled_data[i-n_steps:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# Define the sizes of the training, validation, and testing datasets
train_size = int(0.6 * len(X))
val_size = int(0.2 * len(X))
test_size = len(X) - train_size - val_size

# Split the data into training, validation, and testing sets
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Define the model
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(n_features))

# Compile the model
model_cnn.compile(optimizer='adam', loss='mse')

# Train the model on the training set
model_cnn.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Evaluate the model
y_pred_val = model_cnn.predict(X_val)
y_pred_test = model_cnn.predict(X_test)

# Inverse transform the predictions and actual values
y_pred_val = scaler.inverse_transform(y_pred_val)
y_val = scaler.inverse_transform(y_val)
y_pred_test = scaler.inverse_transform(y_pred_test)
y_test = scaler.inverse_transform(y_test)

# Transform 'SMR20' back to the original scale
y_pred_val[:, data.columns.get_loc('SMR20')] = np.exp(y_pred_val[:, data.columns.get_loc('SMR20')])
y_val[:, data.columns.get_loc('SMR20')] = np.exp(y_val[:, data.columns.get_loc('SMR20')])
y_pred_test[:, data.columns.get_loc('SMR20')] = np.exp(y_pred_test[:, data.columns.get_loc('SMR20')])
y_test[:, data.columns.get_loc('SMR20')] = np.exp(y_test[:, data.columns.get_loc('SMR20')])

# Calculate evaluation metrics
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
print('Validation Set Metrics:')
print('MAE:', mae_val)
print('MSE:', mse_val)
print('RMSE:', rmse_val)
print('R^2:', r2_val)
print('MAPE:', mape_val)
print('Testing Set Metrics:')
print('MAE:', mae_test)
print('MSE:', mse_test)
print('RMSE:', rmse_test)
print('R^2:', r2_test)
print('MAPE:', mape_test)