# Phishing Email Detection System

High-accuracy, production-ready machine learning pipeline for detecting phishing emails using the unified `phishing_email.csv` dataset (82,482 samples).

---

## 1. Overview & Architecture

The system processes raw email text, extracts sublinear TF-IDF word and bi-gram features, and feeds them into calibrated probabilistic classifiers.

- **Primary Production Model**: Weighted Soft-Voting Ensemble combining **SGDClassifier (Modified Huber loss)**, **Logistic Regression**, and **Multinomial Naive Bayes**.
- **Alternative Fast Single Model**: **SGDClassifier (Modified Huber)** (`models/phishing_detector_sgd.joblib`).
- **Feature Extraction**: TF-IDF with 75,000 max features, n-gram range `(1, 2)` (unigrams & bigrams), sublinear term frequency scaling, and unicode accent normalization.
- **Explainability**: Every prediction includes local token attribution scores showing the top keywords that triggered the classification.

---

## 2. Benchmark Results on Test Set (16,497 emails)

Evaluated on a held-out 20% stratified test split:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SGDClassifier (ModHuber)** | **99.28%** | 99.23% | **99.39%** | **99.31%** | 0.9993 | 43.6s |
| **Ensemble (Production)** | **99.21%** | **99.31%** | 99.17% | 99.24% | **0.9996** | 46.2s |
| **Logistic Regression** | 99.08% | 99.09% | 99.15% | 99.12% | 0.9995 | 45.5s |
| **Multinomial Naive Bayes** | 97.85% | 99.55% | 96.29% | 97.90% | 0.9988 | 44.8s |

### Confusion Matrix (Ensemble Model at 0.50 Threshold)
- **True Negatives (Legitimate correctly identified)**: 7,860
- **False Positives (Legitimate flagged as phishing)**: 59 (0.75%)
- **False Negatives (Missed phishing)**: 71 (0.83%)
- **True Positives (Phishing correctly caught)**: 8,507 (99.17%)

---

## 3. Project Structure

```
├── Email_phishing/
│   ├── phishing_email.csv       # Unified dataset (82,486 rows)
│   ├── CEAS_08.csv              # Source component dataset
│   ├── Enron.csv                # Source component dataset
│   ├── Ling.csv                 # Source component dataset
│   ├── Nazario.csv              # Source component dataset
│   ├── Nigerian_Fraud.csv       # Source component dataset
│   └── SpamAssasin.csv          # Source component dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Robust ingestion, validation & split
│   └── model.py                 # PhishingDetector class & explanations
├── models/
│   ├── phishing_detector.joblib     # Production ensemble model
│   ├── phishing_detector_sgd.joblib # Production fast SGD model
│   └── metrics.json                 # Comprehensive benchmark metrics
├── train.py                     # Training and evaluation script
├── predict.py                   # Prediction CLI & Python API
├── demo.py                      # Automated validation suite (6 scenarios)
└── README.md
```

---

## 4. Usage Instructions

### REST API Endpoints & Python Client

The backend exposes a full CORS-enabled REST API:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Health check, model status, and service metadata |
| `/api/predict` | `POST` | Core phishing inference, threat scoring, and keyword attribution |
| `/api/batch` | `POST` | Bulk email inspection with latency metrics |
| `/api/metrics` | `GET` | Model evaluation benchmarks & confusion matrix |
| `/api/samples` | `GET` | Pre-configured realistic test emails |

#### Using the Python API Client (`api_client.py`)
```python
from api_client import PhishGuardClient

client = PhishGuardClient("http://127.0.0.1:5000")

# Check backend status
print(client.health())

# Analyze an email via REST API
result = client.predict(
    text="URGENT: Your account has been suspended. Click here to verify http://fake-login.com",
    subject="Account Alert",
    sender="security@bank-alert.com"
)
print("Verdict:", result["prediction"])
print("Phishing Probability:", result["phishing_probability"])
print("Top Triggers:", result["top_phishing_indicators"])
```

#### Run API Integration Tests
```bash
python test_api_connection.py
```

### Run the Web Interface (Dashboard)
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser to access:
- **Live Email Inspector**: Real-time phishing threat scoring with animated circular gauge, risk tiers, and keyword highlightings.
- **Sample Presets**: Quick 1-click test scenarios for both phishing attacks and legitimate business emails.
- **Batch Scanner**: Paste or upload multiple emails separated by `---` for bulk analysis and CSV export.
- **Model Analytics**: Live performance benchmarks, confusion matrix, and training dataset composition.
- **REST API Explorer**: Interactive documentation and cURL / Python client snippets.

### Run Inference via CLI
```bash
# Analyze a single email string
python predict.py -t "URGENT: Your account has been suspended. Click here to verify your credentials."

# Analyze an email file
python predict.py -f path/to/email.txt

# Launch interactive analysis shell
python predict.py -i
```

### Run Automated Demo Scenarios
```bash
python demo.py
```

### Retrain Models
```bash
python train.py
```

### Python API Example
```python
from predict import analyze_email

email_text = "Dear customer, your bank account is locked. Please verify immediately."
result = analyze_email(email_text)

print(f"Prediction: {result['prediction']}")
print(f"Phishing Probability: {result['phishing_probability'] * 100:.2f}%")
print(f"Risk Level: {result['risk_level']}")
print("Top Phishing Triggers:", result['top_phishing_indicators'])
```
