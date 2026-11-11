# PhishGuard AI — Complete Step-by-Step Execution Guide

This document provides a structured, step-by-step walkthrough to run, test, and develop every component of the **Phishing Email Detection System**.

---

## Table of Contents
1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Step 1: Launch Web Dashboard & REST API](#2-step-1-launch-web-dashboard--rest-api)
3. [Step 2: Run Automated Validation Suite (Demo)](#3-step-2-run-automated-validation-suite-demo)
4. [Step 3: Run API Integration Tests](#4-step-3-run-api-integration-tests)
5. [Step 4: Run Predictions via CLI](#5-step-4-run-predictions-via-cli)
6. [Step 5: Programmatic Integration (Python Client)](#6-step-5-programmatic-integration-python-client)
7. [Step 6: Model Training & Benchmarking Pipeline](#7-step-6-model-training--benchmarking-pipeline)
8. [Project File Map & Quick Reference](#8-project-file-map--quick-reference)

---

## 1. Prerequisites & Environment Setup

### Terminal Navigation
Open PowerShell or your preferred command line and navigate to the project root:
```powershell
cd "d:\CODING\Phishing Email Detection System"
```

### Python Dependencies
Ensure Python 3.10+ is installed along with the required libraries:
```powershell
pip install flask scikit-learn numpy joblib
```

---

## 2. Step 1: Launch Web Dashboard & REST API

The web application exposes a full-featured browser dashboard and high-performance REST API.

### Start the Server:
```powershell
python app.py
```

### Access URLs:
* **Web Dashboard**: [http://127.0.0.1:5000](http://127.0.0.1:5000) (or [http://localhost:5000](http://localhost:5000))
* **Health Check API**: [http://127.0.0.1:5000/api/health](http://127.0.0.1:5000/api/health)
* **Metrics API**: [http://127.0.0.1:5000/api/metrics](http://127.0.0.1:5000/api/metrics)
* **Samples API**: [http://127.0.0.1:5000/api/samples](http://127.0.0.1:5000/api/samples)

### Dashboard Capabilities:
* **Live Inspector**: Enter sender, subject, and body text. Click **"Analyze Threat via API"** (or press <kbd>Ctrl+Enter</kbd>) to receive real-time probability gauge scores, risk tiers, and highlighted keyword attribution clues.
* **Sample Presets**: Quick 1-click loading of realistic phishing attacks and legitimate business emails.
* **Batch Scanner**: Paste multiple emails separated by `---` for bulk threat triage and CSV export.
* **Model Analytics**: Live view of precision, recall, F1, ROC-AUC, and the test confusion matrix.
* **API Integration**: Test live endpoints directly with the interactive API Playground.

> **Tip**: Press <kbd>Ctrl</kbd> + <kbd>C</kbd> in the terminal running `app.py` to stop the server.

---

## 3. Step 2: Run Automated Validation Suite (Demo)

Validate that the production model correctly classifies diverse phishing techniques and legitimate workplace communications:

```powershell
python demo.py
```

### What This Tests:
* Scenario 1: Urgent Account Restriction lure (*Phishing*)
* Scenario 2: Advance fee wire transfer fraud (*Phishing*)
* Scenario 3: IT helpdesk credential harvesting (*Phishing*)
* Scenario 4: Engineering team meeting reschedule (*Legitimate*)
* Scenario 5: Open source pull request review (*Legitimate*)
* Scenario 6: Enterprise vendor purchase order (*Legitimate*)

---

## 4. Step 3: Run API Integration Tests

Verify that all backend endpoints, serialization, and status codes are functioning with 100% pass rate:

```powershell
python test_api_connection.py
```

### Test Scope:
1. `GET /api/health` — Verifies service status and preloaded model readiness.
2. `POST /api/predict` (Phishing) — Tests classification and probability thresholding.
3. `POST /api/predict` (Legitimate) — Tests false-positive suppression on benign emails.
4. `POST /api/batch` — Verifies bulk processing and latency measurements.
5. `GET /api/metrics` & `GET /api/samples` — Validates evaluation metadata.

*Note: The test script automatically detects whether the live HTTP server is running or falls back to an internal test client.*

---

## 5. Step 4: Run Predictions via CLI

Use [`predict.py`](predict.py) for fast terminal-based inspection without needing the web UI.

### Option A: Analyze Single Email String
```powershell
python predict.py -t "URGENT: Your account access has been restricted. Click here to verify your identity http://login-verify.com"
```

### Option B: Analyze Email Saved in File
```powershell
python predict.py -f "path/to/email.txt"
```

### Option C: Interactive Prompt Shell
```powershell
python predict.py -i
```
*Type or paste email text directly into the prompt. Type `exit` or leave blank to quit.*

---

## 6. Step 5: Programmatic Integration (Python Client)

### Direct Local Inference (No Server Required)
```python
from predict import analyze_email

email_text = "URGENT: Please verify your domain credentials immediately."
result = analyze_email(email_text)

print("Classification:", result["prediction"])
print(f"Phishing Probability: {result['phishing_probability'] * 100:.2f}%")
print("Risk Level:", result["risk_level"])
print("Top Indicators:", result["top_phishing_indicators"])
```

### Remote REST API Client ([`api_client.py`](api_client.py))
```python
from api_client import PhishGuardClient

client = PhishGuardClient("http://127.0.0.1:5000")

# 1. Single email check
res = client.predict(
    text="Your corporate mailbox quota has exceeded 99%. Update now.",
    subject="Action Required",
    sender="admin@corp-migration.net"
)
print("Verdict:", res["prediction"], f"({res['phishing_probability']*100:.1f}%)")

# 2. Batch check
batch_res = client.predict_batch([
    "Urgent security notice: verify account now",
    "Can we reschedule our architecture sync to Friday at 2pm?"
])
for item in batch_res["results"]:
    print(f"#{item['index']}: {item['prediction']} ({item['phishing_probability']*100:.1f}%)")
```

---

## 7. Step 6: Model Training & Benchmarking Pipeline

The repository includes pre-trained production model weights:
* `models/phishing_detector.joblib` — Primary Soft-Voting Ensemble (SGD + Logistic + Naive Bayes)
* `models/phishing_detector_sgd.joblib` — Lightweight fast SGD model
* `models/metrics.json` — Comprehensive benchmark evaluation on 16,497 test samples

### Inspect Current Production Model Status:
```powershell
python train.py
```

### Test-Run Training Pipeline on Bootstrap Samples:
```powershell
python train.py --demo-train
```
*Generates a balanced dataset, trains and benchmarks all 4 architectures (SGD, Logistic, Naive Bayes, Ensemble), finds the optimal decision threshold, and saves to `models/phishing_detector_demo.joblib` without touching production weights.*

### Retrain on a Custom CSV Dataset:
```powershell
python train.py --dataset "path/to/your_dataset.csv" --force
```

---

## 8. Project File Map & Quick Reference

### Directory Structure
```
d:\CODING\Phishing Email Detection System\
│
├── app.py                     # Flask Web Application & REST API backend
├── train.py                   # Model training, benchmarking & threshold optimization
├── predict.py                 # CLI prediction tool & Python inference API
├── demo.py                    # Automated validation test suite (6 scenarios)
├── test_api_connection.py     # End-to-end API integration tests
├── api_client.py              # Lightweight Python REST client wrapper
├── HOW_TO_RUN.md              # This execution and operation guide
│
├── models/
│   ├── phishing_detector.joblib     # Production Soft-Voting Ensemble model (3.75 MB)
│   ├── phishing_detector_sgd.joblib # Production Fast SGD model (1.42 MB)
│   └── metrics.json                 # Evaluation metrics on 16,497 test emails
│
├── src/
│   ├── __init__.py            # Package initializer
│   ├── model.py               # PhishingDetector architecture & explainability
│   └── data_loader.py         # CSV ingestion, data validation & train/test split
│
├── templates/
│   └── index.html             # Flask Jinja2 web dashboard template
└── frontend/
    └── index.html             # Standalone / static web dashboard
```

### Quick Commands Cheat Sheet
| Task | Command |
| :--- | :--- |
| **Start Web Server** | `python app.py` |
| **Open Browser UI** | Navigate to `http://127.0.0.1:5000` |
| **Run Scenario Tests** | `python demo.py` |
| **Run API Tests** | `python test_api_connection.py` |
| **Interactive Terminal Shell** | `python predict.py -i` |
| **Analyze Text via CLI** | `python predict.py -t "email text"` |
| **Pipeline Training Demo** | `python train.py --demo-train` |
