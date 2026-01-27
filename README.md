# Anomaly Detection System

A machine learning system for detecting anomalies in infrastructure and application metrics using LSTM, LSTM Autoencoder, and TCN models.

## Features
- Three specialized deep learning models
- CSV data processing pipeline
- Customizable configuration
- Anomaly detection and prediction modes
- Training and evaluation framework

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation
The system processes CSV files containing feature-engineered metrics from:
- Application logs
- Infrastructure metrics
- Distributed tracing data

Place your data files in the `data/` directory:
- `train.csv` - Training dataset
- `test.csv` - Evaluation dataset

## Model Training

### LSTM Model
```bash
python3 -m lstm.train
```

### LSTM Autoencoder
```bash
python3 -m lstm_autoencoder.train
```

### Temporal Convolutional Network (TCN)
```bash
python3 -m tcn.train
```

## Anomaly Detection & Prediction

### Detect anomalies in current data:
```bash
python3 anomaly_detect.py
```

### Run predictions:
```bash
python3 predict.py
```

## Configuration
Modify model parameters and training settings in:
`config/config.py`

Key configurable parameters:
- Sequence length
- Batch size
- Epochs
- Learning rate
- Threshold sensitivity

## Project Structure
```
aiPoc/
├── common/          # Shared utilities
├── config/          # Configuration files
├── data/            # Datasets (CSV format)
├── lstm/            # LSTM model implementation
├── lstm_autoencoder/ # Autoencoder model
├── tcn/             # Temporal Convolutional Network
├── models/          # Trained model checkpoints
├── logs/            # Training logs
├── requirements.txt # Python dependencies
├── anomaly_detect.py # Main detection script
└── predict.py       # Prediction script
```

## Dependencies
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

See `requirements.txt` for complete list.

## License
[MIT License](LICENSE)