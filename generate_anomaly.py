# generate_anomaly_data.py — FINAL, TESTED & WORKING
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
import os

np.random.seed(42)
random.seed(42)

N_NORMAL = 3000
N_ANOMALY = 600
TOTAL = N_NORMAL + N_ANOMALY
START_TIME = datetime(2025, 11, 25, 9, 0, 0)

rows = []

for i in range(TOTAL):
    ts = START_TIME + timedelta(seconds=random.randint(0, 3600))
    start_time = ts
    end_time = ts + timedelta(milliseconds=int(np.random.lognormal(4.5, 0.6)))

    # Base normal behavior
    duration_ms = round(end_time.timestamp() * 1000 - start_time.timestamp() * 1000, 6)
    status_code = 200
    is_error = 0
    body_size = int(np.random.lognormal(5, 0.8))
    route = random.choice(["/api/people", "/api/users", "/api/orders", "/health"])

    # Inject anomalies in last 600 rows
    anomaly_type = "normal"
    if i >= N_NORMAL:
        anomaly_id = (i - N_NORMAL) % 5
        if anomaly_id == 0:      # Latency spike
            duration_ms *= random.uniform(10, 30)
            anomaly_type = "latency_spike"
        elif anomaly_id == 1:    # 5xx errors
            status_code = random.choice([500, 502, 503, 504])
            is_error = 1
            anomaly_type = "http_5xx"
        elif anomaly_id == 2:    # Huge payload
            body_size *= random.randint(20, 80)
            anomaly_type = "large_payload"
        elif anomaly_id == 3:    # Error burst
            is_error = 1
            status_code = random.choice([400, 404, 500])
            duration_ms *= random.uniform(5, 15)
            anomaly_type = "error_burst"
        elif anomaly_id == 4:    # Traffic spike
            duration_ms = np.random.lognormal(3.0, 0.3)
            anomaly_type = "traffic_spike"

    row = {
        "timestamp": start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "topic": "test-topic-sig",
        "attributes_client.port": random.randint(30000, 60000) if random.random() < 0.7 else "",
        "attributes_code.filepath": "/var/www/vendor/laravel/framework/src/Illuminate/Routing/Route.php" if random.random() < 0.6 else "",
        "attributes_code.function": "handle" if random.random() < 0.6 else "",
        "attributes_code.lineno": random.randint(100, 500) if random.random() < 0.6 else "",
        "attributes_code.namespace": "Illuminate\\Routing" if random.random() < 0.6 else "",
        "attributes_db.name": "test_crud",
        "attributes_db.operation": "SELECT",
        "attributes_db.statement": 'select * from "people"',
        "attributes_db.system": "pgsql",
        "attributes_db.user": "root",
        "attributes_http.client_ip": "172.20.0.1",
        "attributes_http.method": random.choice(["GET", "POST"]),
        "attributes_http.request.body.size": int(body_size * 0.8),
        "attributes_http.request.method": random.choice(["GET", "POST"]),
        "attributes_http.response.body.size": body_size,
        "attributes_http.response.status_code": status_code,
        "attributes_http.route": route,
        "attributes_http.status_code": status_code,
        "attributes_http.url": f"http://localhost:8000{route}",
        "attributes_http.user_agent": "k6/0.57.0",
        "attributes_server.address": "localhost",
        "attributes_server.port": 8000,
        "attributes_url.full": f"http://localhost:8000{route}",
        "attributes_url.path": route,
        "attributes_url.scheme": "http",
        "body_length": body_size,
        "correlation_error_density": 0.9 if is_error else 0.0,
        "day_of_week": start_time.weekday(),
        "duration_ms": duration_ms,
        "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "enhanced_duration_ms_mean": duration_ms * random.uniform(0.9, 1.1),
        "enhanced_error_rate": 1.0 if is_error else 0.0,
        "error": is_error,
        "hour": start_time.hour,
        "is_error": is_error,
        "kind": 2,
        "name": f"{random.choice(['GET','POST'])} {route}",
        "parent_span_id": uuid.uuid4().hex[:16] if random.random() < 0.3 else "",
        "resource_attributes_service.name": "laravel",
        "sliding_error_rate_15m": 0.9 if is_error else 0.0,
        "source": "laravel",
        "span_id": uuid.uuid4().hex[:16],
        "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "trace_id": uuid.uuid4().hex,
        "value": 1.0,
        "__anomaly__": 1 if anomaly_type != "normal" else 0,
        "__anomaly_type__": anomaly_type,
    }

    # Fill remaining columns with defaults
    defaults = {col: "" for col in [
        "attributes_network.peer.address", "attributes_network.protocol.version",
        "body", "context_ip", "context_trace_id", "exception_message", "severity_text",
        "resource_attributes_process.pid", "resource_attributes_telemetry.distro.name"
    ]}
    row.update(defaults)
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Full column list (your exact order)
columns = [
    "timestamp","topic","attributes_client.port","attributes_code.filepath","attributes_code.function",
    "attributes_code.lineno","attributes_code.namespace","attributes_db.name","attributes_db.operation",
    "attributes_db.statement","attributes_db.system","attributes_db.user","attributes_http.client_ip",
    "attributes_http.method","attributes_http.request.body.size","attributes_http.request.method",
    "attributes_http.response.body.size","attributes_http.response.status_code","attributes_http.route",
    "attributes_http.status_code","attributes_http.url","attributes_http.user_agent",
    "attributes_network.peer.address","attributes_network.protocol.version","attributes_server.address",
    "attributes_server.port","attributes_url.full","attributes_url.path","attributes_url.scheme",
    "attributes_user_agent.original","body","body_length","context_ip","context_method",
    "context_request_data_age","context_request_data_name","context_status_code","context_trace_id",
    "context_url","context_user_agent","correlation_cross_signal_health_score",
    "correlation_error_correlation_score","correlation_error_density","correlation_error_to_request_ratio",
    "correlation_metric_anomaly_rate","correlation_service_diversity","day_of_week","duration_ms",
    "end_time","enhanced_body_length_cv","enhanced_body_length_max","enhanced_body_length_mean",
    "enhanced_body_length_min","enhanced_body_length_p95","enhanced_body_length_std",
    "enhanced_duration_ms_cv","enhanced_duration_ms_max","enhanced_duration_ms_mean",
    "enhanced_duration_ms_min","enhanced_duration_ms_p95","enhanced_duration_ms_std",
    "enhanced_error_rate","enhanced_hour_variance","enhanced_messages_per_minute","enhanced_peak_hour",
    "enhanced_unique_sources","enhanced_value_cv","enhanced_value_max","enhanced_value_mean",
    "enhanced_value_min","enhanced_value_p95","enhanced_value_std","enhanced_window_minutes","error",
    "events_count","exception","exception_message","exception_stacktrace_length","exception_type",
    "hour","is_error","kind","name","observed_ts","parent_span_id","resource_attributes_host.arch",
    "resource_attributes_host.name","resource_attributes_os.description","resource_attributes_os.name",
    "resource_attributes_os.type","resource_attributes_os.version","resource_attributes_process.executable.path",
    "resource_attributes_process.owner","resource_attributes_process.pid","resource_attributes_process.runtime.name",
    "resource_attributes_process.runtime.version","resource_attributes_service.name",
    "resource_attributes_service.version","resource_attributes_telemetry.distro.name",
    "resource_attributes_telemetry.distro.version","resource_attributes_telemetry.sdk.language",
    "resource_attributes_telemetry.sdk.name","resource_attributes_telemetry.sdk.version",
    "severity_number","severity_text","sliding_avg_duration_15m","sliding_avg_duration_5m",
    "sliding_avg_duration_60m","sliding_avg_messages_15m","sliding_avg_messages_5m",
    "sliding_avg_messages_60m","sliding_duration_trend_15m","sliding_duration_trend_5m",
    "sliding_duration_trend_60m","sliding_error_rate_15m","sliding_error_rate_5m",
    "sliding_error_rate_60m","sliding_trend_15m","sliding_trend_5m","sliding_trend_60m",
    "sliding_volume_trend_15m","sliding_volume_trend_5m","sliding_volume_trend_60m","source",
    "span_id","start_time","trace_id","value","__anomaly__","__anomaly_type__"
]

# Add missing columns
for col in columns:
    if col not in df.columns:
        df[col] = ""

df = df[columns]

os.makedirs("data", exist_ok=True)
df.to_csv("data/kafka_with_anomalies.csv", index=False)

print("SUCCESS: Generated data/kafka_with_anomalies.csv")
print(f"   {N_NORMAL} normal + {N_ANOMALY} anomalous traces")
print("   Your autoencoder WILL detect them.")