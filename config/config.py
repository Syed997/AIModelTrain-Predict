# TODO: need to convert this into .env
INPUT_WINDOW = 25
FORECAST_HORIZON = 10
N_FEATURES = 5
EPOCHS = 5
THRESHOLDS = [5, 5, 5, 5, 5]
STOP_THRESHOLD = 1e-5

BATCH_SIZE = 16
TRAIN_SPLIT = 0.8

TCN_MODEL_FILENAME = "tcn_best.pth"
LSTM_MODEL_FILENAME = "lstm_best.pth"
TRAIN_CSV = "train.csv"
TEST_CSV = "temp.csv"
EVAL_CSV = "eval.csv"
TIMESERIES_MODE = "test"  # "train" or "test"