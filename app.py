"""
Flask Web Application & REST API Backend for Phishing Email Detection.
Features full CORS support, health checking, probabilistic inference,
token attribution, batch processing, and live performance metrics.
"""

import os
import time
import json
from flask import Flask, render_template, request, jsonify
from predict import analyze_email, get_detector

app = Flask(__name__)

# Preload model into memory at startup
detector = None
try:
    detector = get_detector()
    print("Model preloaded successfully into Flask app.")
except Exception as e:
    print(f"Warning: Failed to preload model: {e}")

# Predefined realistic email samples for quick testing in UI
SAMPLE_EMAILS = [
    {
        "id": "paypal_suspension",
        "title": "PayPal Account Restriction",
        "type": "Phishing",
        "sender": "service@security-paypal-alert.com",
        "subject": "URGENT: Your account access has been limited",
        "body": "URGENT SECURITY NOTICE: Your PayPal account has been temporarily restricted due to suspicious login attempts from an unrecognized device.\n\nPlease verify your identity immediately by clicking on the secure link below to avoid permanent suspension:\nhttp://secure-paypal-verification-update.com/login?id=83921\n\nFailure to verify within 24 hours will result in permanent account termination and forfeiture of funds."
    },
    {
        "id": "nigerian_scam",
        "title": "Inheritance Wire Transfer",
        "type": "Phishing",
        "sender": "barrister.musa@lawchambers-intl.org",
        "subject": "Confidential Business Partnership - $25.5M USD",
        "body": "DEAR BELOVED FRIEND,\n\nI am Barrister Mohammed Musa, personal attorney to the late Dr. George who tragically died with his family. He left behind an unclaimed sum of $25.5 Million USD in a secure escrow vault.\n\nI am seeking your urgent assistance to stand as the foreign beneficiary so these funds can be wired to your bank account. You will receive 40% of the total funds. Please reply with your full legal name, private telephone number, and bank details immediately."
    },
    {
        "id": "it_helpdesk",
        "title": "Corporate IT Password Reset",
        "type": "Phishing",
        "sender": "helpdesk-admin@server-migration-it.net",
        "subject": "Action Required: Email Quota Exceeded (99% Full)",
        "body": "ATTENTION ALL EMPLOYEES:\n\nOur IT Helpdesk is performing a critical email server migration this weekend. Your corporate mailbox has exceeded 99% storage quota.\n\nTo prevent immediate suspension of email service and loss of incoming communications, update your domain credentials now:\nhttp://it-support-portal-webmail-update.org/auth\n\nIT Support Desk & System Administration"
    },
    {
        "id": "meeting_sync",
        "title": "Engineering Team Sync",
        "type": "Legitimate",
        "sender": "david.chen@enterprise.tech",
        "subject": "Rescheduling our bi-weekly architecture sync",
        "body": "Hi Sarah,\n\nCould we reschedule our bi-weekly architecture sync from Thursday morning to Friday at 2:00 PM? I need to finalize the quarterly performance metrics and review the API latency logs before we meet.\n\nLet me know if that time works for you.\n\nBest regards,\nDavid Chen\nSoftware Engineering Lead"
    },
    {
        "id": "code_review",
        "title": "Pull Request Review",
        "type": "Legitimate",
        "sender": "alex.m@github-enterprise.internal",
        "subject": "Review for PR #142: Database connection pooling fix",
        "body": "Thanks for submitting pull request #142!\n\nI reviewed the commit diff for the database connection pooling fix. Overall the logic looks clean, but could you add a unit test covering timeout retries under network partition?\n\nOnce the CI tests pass on Linux and Windows, I will approve and merge.\n\nCheers,\nAlex"
    },
    {
        "id": "invoice_po",
        "title": "Vendor Purchase Order",
        "type": "Legitimate",
        "sender": "billing@cloudservices-global.com",
        "subject": "Approved Purchase Order #88412 - Cloud Renewal",
        "body": "Good morning Robert,\n\nAttached is the approved purchase order #88412 for the cloud infrastructure renewal for the upcoming fiscal quarter.\n\nPlease confirm receipt and let us know the estimated provisioning timeline.\n\nThank you for your partnership.\nAccounts Payable Department"
    }
]


# =========================================================================
# CORS & PREFLIGHT HEADERS (Enables any external or local frontend to connect)
# =========================================================================
@app.before_request
def handle_preflight():
    """Handles HTTP OPTIONS preflight requests for cross-origin frontend clients."""
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        return response


@app.after_request
def add_cors_headers(response):
    """Appends CORS headers to all outgoing responses."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response


# =========================================================================
# WEB FRONTEND ROUTE
# =========================================================================
@app.route("/")
def index():
    """Renders the dashboard UI."""
    return render_template("index.html", samples=SAMPLE_EMAILS)


# =========================================================================
# REST API ENDPOINTS
# =========================================================================
@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint to verify backend status, model readiness, and API latency."""
    return jsonify({
        "status": "healthy",
        "service": "PhishGuard AI Backend API",
        "version": "2.0.0",
        "model_loaded": detector is not None,
        "model_type": getattr(detector, "model_type", "ensemble"),
        "timestamp": time.time(),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Core prediction endpoint.
    Accepts JSON:
      - Option A: {"text": "full email content..."}
      - Option B: {"sender": "...", "subject": "...", "body": "..."}
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid request body. Expected JSON payload."}), 400

    raw_text = data.get("text", "")
    subject = data.get("subject", "").strip()
    sender = data.get("sender", "").strip()
    body = data.get("body", "").strip()

    # Construct combined text if separate fields are passed
    if not raw_text:
        parts = []
        if sender:
            parts.append(f"From: {sender}")
        if subject:
            parts.append(f"Subject: {subject}")
        if body:
            parts.append(body)
        raw_text = " ".join(parts)

    if not raw_text.strip():
        return jsonify({"error": "Email text cannot be empty."}), 400

    t_start = time.time()
    result = analyze_email(raw_text)
    latency_ms = round((time.time() - t_start) * 1000, 2)

    result["input_length"] = len(raw_text)
    result["inference_latency_ms"] = latency_ms
    return jsonify(result)


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """
    Batch prediction endpoint.
    Accepts JSON: {"emails": ["email 1", "email 2", ...]}
    """
    data = request.get_json(force=True, silent=True)
    if not data or "emails" not in data or not isinstance(data["emails"], list):
        return jsonify({"error": "Expected JSON with an 'emails' list."}), 400

    emails = data["emails"][:100]  # Cap at 100 per batch for web responsiveness
    t_start = time.time()
    results = []

    for idx, em in enumerate(emails):
        if isinstance(em, dict):
            text = f"{em.get('subject', '')} {em.get('body', '')}".strip() or em.get("text", "")
        else:
            text = str(em)

        res = analyze_email(text)
        results.append({
            "index": idx + 1,
            "snippet": text[:80] + ("..." if len(text) > 80 else ""),
            "prediction": res["prediction"],
            "phishing_probability": res["phishing_probability"],
            "risk_level": res["risk_level"]
        })

    total_latency_ms = round((time.time() - t_start) * 1000, 2)
    return jsonify({
        "total": len(results),
        "results": results,
        "batch_latency_ms": total_latency_ms,
        "avg_per_email_ms": round(total_latency_ms / len(results), 2) if results else 0
    })


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns model benchmark metrics from metrics.json."""
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "metrics.json not found"}), 404


@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Returns the library of test sample emails."""
    return jsonify(SAMPLE_EMAILS)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"\n========================================================")
    print(f"  PHISHING EMAIL DETECTOR BACKEND API SERVER")
    print(f"  Base URL: http://{host}:{port}")
    print(f"  Health Check: http://{host}:{port}/api/health")
    print(f"  Predict API:  http://{host}:{port}/api/predict")
    print(f"========================================================\n")
    app.run(host=host, port=port, debug=False)
