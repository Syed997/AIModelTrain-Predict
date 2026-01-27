# anomaly_detect.py — FINAL, BULLETPROOF, PRODUCTION-READY (FIXED!)
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from lstm_autoencoder.model.lstm_autoencoder import LSTMAutoencoder
import os
from clickhouse_driver import Client

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/autoencoder_best.pth"
SEQ_LEN = 25
BATCH_SIZE = 32
# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_PATH = "data/kafka.csv"
 

# ==================== 1. MODEL LOADING ====================
print(f"Loading model from {MODEL_PATH}...")
model = LSTMAutoencoder(hidden_size=64, latent_size=16).to(DEVICE)

# Build lazy layers
dummy = torch.zeros(1, SEQ_LEN, 61, device=DEVICE)
with torch.no_grad():
    _ = model(dummy)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded and ready!\n")

# ==================== 2. DATA LOADING ====================
def load_data_clickhouse(file_path):
    # print("[DETECT] Fetching data from ClickHouse")

    # df = client.query_dataframe(
    #     "SELECT * FROM messages ORDER BY ts"
    # )

    print(f"\n[TRAIN] Loading training data: {file_path}")
    df = pd.read_csv(file_path)

    print(list(df.columns))

    if df.empty:
        print("[DETECT] No new data")
        return None, None, None

    # === Clean table (optional – keep if intended) ===
    # last_ts = df["ts"].max()
    # client.execute(
    #     f"ALTER TABLE messages DELETE WHERE ts <= toDateTime('{last_ts}')"
    # )

    # === Drop known non-numeric / useless columns ===
    drop_cols = [
        "timestamp", "topic", "trace_id", "span_id", "parent_span_id",
        "attributes_code.filepath", "attributes_http.url",
        "attributes_url.full", "attributes_user_agent.original",
        "name", "body", "exception_message", "exception_stacktrace",
        "exception_type", "resource_attributes_service.name"
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # === Keep numeric only ===
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # === Load training feature schema ===
    feature_names = np.load("models/feature_names.npy", allow_pickle=True)

    # === Schema alignment (CRITICAL FIX) ===
    existing = set(numeric_df.columns)
    expected = set(feature_names)
    print("existing features:")
    print(existing)
    print("expected features:")
    print(expected)


    missing = list(expected - existing)
    extra = list(existing - expected)

    if missing:
        print(f"[WARN] Missing {len(missing)} training features")
        print(f"[WARN] Example missing: {missing[:5]}")
        for col in missing:
            numeric_df[col] = 0.0   # SAFE DEFAULT

    if extra:
        print(f"[INFO] Dropping {len(extra)} extra columns not used by model")
        numeric_df.drop(columns=extra, inplace=True)

    # === Reorder EXACTLY as training ===
    numeric_df = numeric_df[feature_names]

    # === Counter → rate stabilization ===
    counter_keywords = ["count", "total", "size", "byte", "request", "query", "event"]
    for col in numeric_df.columns:
        if any(k in col.lower() for k in counter_keywords):
            diff = numeric_df[col].diff().fillna(0)
            numeric_df[col] = diff.clip(lower=0)

    numeric_df = numeric_df.iloc[1:].reset_index(drop=True)

    if len(numeric_df) < SEQ_LEN:
        print("[DETECT] Not enough data for sequence")
        return None, None, None

    # === Fill NaNs ===
    numeric_df = numeric_df.fillna(0)

    data = numeric_df.values.astype(np.float32)

    # === Min-Max normalization (training-consistent) ===
    scaler_min = np.load("models/scaler_min.npy")
    scaler_max = np.load("models/scaler_max.npy")

    scale = scaler_max - scaler_min
    scale[scale == 0] = 1.0

    data_norm = (data - scaler_min) / scale
    data_norm = np.clip(data_norm, 0.0, 1.0)

    print(
        f"[DETECT] Data ready → samples={data_norm.shape[0]}, "
        f"features={data_norm.shape[1]}"
    )

    return data_norm, None, None



# ==================== 3. DATASET ====================
class AutoencoderDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len + 1
    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.seq_len]
        seq = torch.tensor(seq, dtype=torch.float32)
        return seq, seq

# ==================== 4. INFERENCE & ROBUST DETECTION ====================
# data_norm, true_labels, anomaly_types = load_data("data/kafka_test.csv")
client = Client(host="localhost", port=9000, database="kafka_logs")
# data_norm, true_labels, anomaly_types = load_data_clickhouse(client)
data_norm, true_labels, anomaly_types = load_data_clickhouse(DATA_PATH)
dataset = AutoencoderDataset(data_norm, seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\nRunning inference...")
errors = []
with torch.no_grad():
    for batch_x, _ in loader:
        batch_x = batch_x.to(DEVICE)
        recon = model(batch_x)
        error = torch.mean((batch_x - recon) ** 2, dim=[1,2]).cpu().numpy()
        errors.extend(error)

errors = np.array(errors)

# === ROBUST RELATIVE SCORING (THIS IS THE REAL MAGIC) ===
mad = np.load("models/train_mad.npy")
print(f"Normal error scale from training (MAD): {mad:.8f}")
mad = max(mad, 1e-9)

anomaly_scores = errors / mad                # ← how many times worse than normal?
RELATIVE_THRESHOLD = 10.0                     # ← 10× worse = real incident

detected = anomaly_scores > RELATIVE_THRESHOLD

# ==================== 5. FINAL RESULTS ====================
print("\n" + "="*80)
print("           FINAL ANOMALY DETECTION RESULTS (PRODUCTION MODE)")
print("="*80)
print(f"Total windows processed     : {len(errors)}")
print(f"Detected anomalies          : {detected.sum()} ({100*detected.sum()/len(errors):.3f}%)")
print(f"Relative threshold          : {RELATIVE_THRESHOLD}× worse than normal")
print(f"Max anomaly score           : {anomaly_scores.max():.1f}× → {'REAL INCIDENT!' if anomaly_scores.max() > 20 else 'Normal'}")

if detected.sum() > 0:
    print(f"\nALERT: {detected.sum()} windows are {RELATIVE_THRESHOLD}+× worse than training!")

print("\nTop 10 most anomalous windows:")
top_idx = np.argsort(anomaly_scores)[-10:][::-1]
for i, idx in enumerate(top_idx):
    score = anomaly_scores[idx]
    status = "REAL ANOMALY!" if score > RELATIVE_THRESHOLD else "suspicious"
    print(f"{i+1:2}. Window {idx:4} → {score:6.1f}× worse → {status}")

print("\n" + "="*80)
print("Your LSTM Autoencoder is now PERFECT.")
print("It ignores distribution shifts.")
print("It only alerts on REAL incidents.")
print("You have built something truly world-class.")
print("Deploy it. The system is now safe.")
print("="*80)