import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, concatenate
import matplotlib.pyplot as plt
from datetime import timedelta

DATA_FILE = 'data/preprocessed_data.csv'
PREDICTION_FILE = 'data/predictions.csv'
LOG_TRANSFORM_COLUMN = 'SMR20'
N_STEPS = 3
EPOCHS = 100
BATCH_SIZE = 32
PREDICTION_DAYS = 240
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_STATE = 42  # For reproducibility if needed

def load_and_preprocess_data(file_path, log_transform_col):
    """
    Load data from CSV, apply log transformation, and scale the features.

    Parameters:
    - file_path (str): Path to the CSV file.
    - log_transform_col (str): Column name to apply log transformation.

    Returns:
    - pd.DataFrame: Scaled and transformed DataFrame.
    - MinMaxScaler: Fitted scaler object.
    """
    try:
        # Load data with 'date' parsed as datetime and set as index
        data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        print("Data loaded successfully.")
        
        # Apply log transformation to specified column
        data[log_transform_col] = np.log(data[log_transform_col])
        print(f"Applied log transformation to '{log_transform_col}'.")
        
        # Initialize and fit scaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        print("Data scaling completed using MinMaxScaler.")
        
        return scaled_df, scaler
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {e}")
        raise

def create_sequences(data, n_steps):
    """
    Create input-output sequences for time series forecasting.

    Parameters:
    - data (np.ndarray): Scaled data array.
    - n_steps (int): Number of time steps.

    Returns:
    - np.ndarray: Input sequences.
    - np.ndarray: Output sequences.
    """
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    print(f"Created input-output sequences with {n_steps} time steps.")
    return X, y

def split_data(X, y, train_ratio, val_ratio, test_ratio):
    """
    Split data into training, validation, and testing sets.

    Parameters:
    - X (np.ndarray): Input sequences.
    - y (np.ndarray): Output sequences.
    - train_ratio (float): Proportion of training data.
    - val_ratio (float): Proportion of validation data.
    - test_ratio (float): Proportion of testing data.

    Returns:
    - Tuple[np.ndarray]: Split datasets (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    total = len(X)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    print(f"Data split into Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Testing: {X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_bidirectional_lstm_model(n_steps, n_features):
    """
    Build a Bidirectional LSTM model with a branching architecture.

    Parameters:
    - n_steps (int): Number of time steps.
    - n_features (int): Number of features.

    Returns:
    - keras.Model: Compiled Keras model.
    """
    inputs = Input(shape=(n_steps, n_features))
    
    # Coarse path
    coarse = Bidirectional(LSTM(units=70, activation='relu'))(inputs)
    
    # Fine path
    fine1 = Bidirectional(LSTM(units=16, activation='relu', return_sequences=True))(inputs)
    fine2 = Bidirectional(LSTM(units=8, activation='relu'))(fine1)
    
    # Concatenate paths
    concat = concatenate([coarse, fine2])
    
    # Output layer
    output = Dense(n_features)(concat)
    
    # Define model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    print("Branching Bidirectional LSTM model built and compiled.")
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    """
    Train the Keras model.

    Parameters:
    - model (keras.Model): Compiled Keras model.
    - X_train (np.ndarray): Training input data.
    - y_train (np.ndarray): Training output data.
    - X_val (np.ndarray): Validation input data.
    - y_val (np.ndarray): Validation output data.
    - epochs (int): Number of training epochs.
    - batch_size (int): Training batch size.

    Returns:
    - keras.callbacks.History: Training history.
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, y_val)
    )
    print("Model training completed.")
    return history

def evaluate_model(model, scaler, X_val, y_val, X_test, y_test, data_columns, log_transform_col):
    """
    Evaluate the model on validation and testing datasets.

    Parameters:
    - model (keras.Model): Trained Keras model.
    - scaler (MinMaxScaler): Fitted scaler object.
    - X_val (np.ndarray): Validation input data.
    - y_val (np.ndarray): Validation output data.
    - X_test (np.ndarray): Testing input data.
    - y_test (np.ndarray): Testing output data.
    - data_columns (List[str]): List of column names.
    - log_transform_col (str): Column name that was log-transformed.

    Returns:
    - Dict: Evaluation metrics for validation and testing sets.
    """
    # Predict
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Inverse transform
    y_pred_val = scaler.inverse_transform(y_pred_val)
    y_val = scaler.inverse_transform(y_val)
    y_pred_test = scaler.inverse_transform(y_pred_test)
    y_test = scaler.inverse_transform(y_test)
    
    # Reverse log transformation for specified column
    smr20_idx = data_columns.index(log_transform_col)
    y_pred_val[:, smr20_idx] = np.exp(y_pred_val[:, smr20_idx])
    y_val[:, smr20_idx] = np.exp(y_val[:, smr20_idx])
    y_pred_test[:, smr20_idx] = np.exp(y_pred_test[:, smr20_idx])
    y_test[:, smr20_idx] = np.exp(y_test[:, smr20_idx])
    
    # Calculate metrics
    metrics = {}
    for dataset, (actual, pred) in zip(['Validation', 'Testing'], [(y_val, y_pred_val), (y_test, y_pred_test)]):
        mae = mean_absolute_error(actual, pred)
        mse = mean_squared_error(actual, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, pred)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        metrics[dataset] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R^2': r2,
            'MAPE': mape
        }
    
    # Print metrics
    for dataset in metrics:
        print(f'\n{dataset} Set Metrics:')
        for metric, value in metrics[dataset].items():
            print(f'{metric}: {value:.4f}')
    
    return metrics

def plot_training_history(history):
    """
    Plot the training and validation loss over epochs.

    Parameters:
    - history (keras.callbacks.History): Training history.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_future(model, scaler, data, n_steps, n_features, prediction_days, log_transform_col):
    """
    Predict future values for a specified number of days.

    Parameters:
    - model (keras.Model): Trained Keras model.
    - scaler (MinMaxScaler): Fitted scaler object.
    - data (pd.DataFrame): Original scaled data.
    - n_steps (int): Number of time steps.
    - n_features (int): Number of features.
    - prediction_days (int): Number of future days to predict.
    - log_transform_col (str): Column name that was log-transformed.

    Returns:
    - pd.DataFrame: DataFrame containing future predictions.
    """
    # Get the last n_steps from the scaled data
    last_sequence = data[-n_steps:].values.copy()
    
    predictions = []
    for _ in range(prediction_days):
        # Reshape to match model input
        input_seq = last_sequence.reshape((1, n_steps, n_features))
        # Predict the next value
        pred = model.predict(input_seq)
        # Append prediction to the list
        predictions.append(pred[0])
        # Update the last_sequence by removing the first step and adding the prediction
        last_sequence = np.vstack([last_sequence[1:], pred])
    
    # Convert predictions to numpy array
    predictions = np.array(predictions)
    
    # Inverse transform the predictions
    predictions = scaler.inverse_transform(predictions)
    
    # Reverse log transformation for specified column
    smr20_idx = data.columns.get_loc(log_transform_col)
    predictions[:, smr20_idx] = np.exp(predictions[:, smr20_idx])
    
    # Create date range for future predictions
    last_date = data.index[-1]
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.to_datetime(last_date)
    date_range = pd.date_range(last_date + timedelta(days=1), periods=prediction_days)
    
    # Create DataFrame for predictions
    predictions_df = pd.DataFrame(predictions, columns=data.columns)
    predictions_df['date'] = date_range
    predictions_df = predictions_df.set_index('date')
    
    print(f"Future predictions for the next {prediction_days} days generated.")
    return predictions_df

def save_predictions(predictions_df, file_path):
    """
    Save the predictions DataFrame to a CSV file.

    Parameters:
    - predictions_df (pd.DataFrame): DataFrame containing predictions.
    - file_path (str): Path to save the CSV file.
    """
    try:
        predictions_df.to_csv(file_path)
        print(f"Predictions saved to '{file_path}'.")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        raise

# ------------------------------
# Main Execution Flow
# ------------------------------

def main():
    # Step 1: Load and preprocess data
    scaled_df, scaler = load_and_preprocess_data(DATA_FILE, LOG_TRANSFORM_COLUMN)
    
    # Step 2: Create sequences
    X, y = create_sequences(scaled_df.values, N_STEPS)
    
    # Step 3: Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO
    )
    
    # Step 4: Build model
    n_features = X_train.shape[2]
    model = build_bidirectional_lstm_model(N_STEPS, n_features)
    
    # Step 5: Train model
    history = train_model(model, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)
    
    # Optional: Plot training history
    plot_training_history(history)
    
    # Step 6: Evaluate model
    data_columns = list(scaled_df.columns)
    metrics = evaluate_model(
        model, scaler, X_val, y_val, X_test, y_test, data_columns, LOG_TRANSFORM_COLUMN
    )
    
    # Step 7: Predict future values
    predictions_df = predict_future(
        model, scaler, scaled_df, N_STEPS, n_features, PREDICTION_DAYS, LOG_TRANSFORM_COLUMN
    )
    
    # Step 8: Save predictions
    save_predictions(predictions_df, PREDICTION_FILE)

if __name__ == "__main__":
    main()
