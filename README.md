# Rubber Price Forecasting System

This project is designed to forecast rubber prices(SMR20) using various models, with a focus on the **Branching Bi-LSTM** model for time series prediction. The system includes preprocessing, model training, and prediction components.

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/weiren07/RubberPriceForecastingSystem.git
cd RubberPriceForecastingSystem
```

### Set Up the Virtual Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

### data
Contains data files used in the project:
- `raw_data.csv`: Raw SMR20 rubber price data
- `preprocessed_data.csv`: Data after preprocessing steps
- `model_predictions.csv`: Model predictions for evaluation

### models
Contains scripts for various forecasting models:
- `arima_model.py`: ARIMA model for baseline forecasting
- `lstm_model.py`: Basic LSTM model
- `bidirectional_lstm_model.py`: Bidirectional LSTM model
- `branching_lstm_model.py`: Branching LSTM model
- `branching_bi_lstm_model.py`: Branching Bi-LSTM model, best-performing model
- `cnn_model.py`: CNN model used as a comparison

### utils
Utility scripts:
- `data_quality.py`: Contains functions for data preprocessing and quality checks

## Model Performance

The Branching Bi-LSTM model has been identified as the most effective for this task. Although the performance differences between models are minor with smaller datasets, the Branching Bi-LSTM model shows increasingly significant improvements in accuracy compare to other models as the dataset size grows. 