"""Prometheus metrics for Task 8 (Monitoring & Logging)."""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "cardiorisk_requests_total",
    "Total HTTP requests handled by the API",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "cardiorisk_request_latency_seconds",
    "Request latency in seconds",
    ["path"],
)

PREDICTION_COUNT = Counter(
    "cardiorisk_predictions_total",
    "Number of /predict calls, broken down by predicted class",
    ["risk_label"],
)
