"""
Automated Test Suite for PhishGuard AI Backend API.
Verifies all REST endpoints and connection response formats.
"""

import sys
from api_client import PhishGuardClient


class DirectFlaskClient:
    """In-memory test client wrapper using Flask app directly."""

    def __init__(self):
        from app import app
        self.client = app.test_client()

    def health(self):
        return self.client.get("/api/health").get_json()

    def predict(self, text: str, subject: str = "", sender: str = ""):
        return self.client.post("/api/predict", json={"text": text, "subject": subject, "sender": sender}).get_json()

    def predict_batch(self, emails):
        return self.client.post("/api/batch", json={"emails": emails}).get_json()

    def metrics(self):
        return self.client.get("/api/metrics").get_json()

    def samples(self):
        return self.client.get("/api/samples").get_json()


def run_tests():
    print("=" * 65)
    print("   TESTING BACKEND REST API ENDPOINTS")
    print("=" * 65)

    client = None
    try:
        http_client = PhishGuardClient("http://127.0.0.1:5000")
        http_client.health()
        client = http_client
        print("  [Mode] Connected to live server at http://127.0.0.1:5000")
    except Exception:
        print("  [Mode] Live HTTP server offline. Testing via internal Flask test client...")
        client = DirectFlaskClient()

    # 1. Health check
    print("\n[1/5] Testing GET /api/health...")
    h = client.health()
    assert h.get("status") == "healthy", f"Health status error: {h}"
    print(f"      [PASS] Status: {h['status']} | Model Loaded: {h['model_loaded']} | Version: {h['version']}")

    # 2. Phishing test
    print("\n[2/5] Testing POST /api/predict (Phishing)...")
    res_phish = client.predict(
        text="URGENT: Your account has been suspended. Click here to verify your identity http://scam.com",
        subject="Action Required",
        sender="alert@security-update.com"
    )
    assert res_phish["prediction"] == "Phishing", f"Expected Phishing, got: {res_phish['prediction']}"
    assert res_phish["phishing_probability"] > 0.8, f"Probability too low: {res_phish['phishing_probability']}"
    print(f"      [PASS] Verdict: {res_phish['prediction']} ({res_phish['phishing_probability']*100:.2f}%)")
    print(f"      Triggers: {[t['token'] for t in res_phish['top_phishing_indicators'][:3]]}")

    # 3. Legitimate test
    print("\n[3/5] Testing POST /api/predict (Legitimate)...")
    res_legit = client.predict(
        text="Hi Team, please find attached the meeting minutes and project notes from Friday's architecture review. Thanks, David",
        subject="Weekly Notes",
        sender="david@company.internal"
    )
    assert res_legit["prediction"] == "Legitimate", f"Expected Legitimate, got: {res_legit['prediction']}"
    assert res_legit["phishing_probability"] < 0.3, f"Probability too high: {res_legit['phishing_probability']}"
    print(f"      [PASS] Verdict: {res_legit['prediction']} ({res_legit['legitimate_probability']*100:.2f}% Legitimate)")

    # 4. Batch test
    print("\n[4/5] Testing POST /api/batch...")
    batch = client.predict_batch([
        "URGENT: Click here to claim your inheritance http://wire-transfer.xyz",
        "Can we reschedule our one-on-one meeting to Thursday at 2pm? Thanks"
    ])
    assert batch["total"] == 2, f"Expected 2 results, got {batch['total']}"
    print(f"      [PASS] Processed {batch['total']} emails in {batch['batch_latency_ms']}ms")
    for r in batch["results"]:
        print(f"        #{r['index']}: {r['prediction']} ({r['phishing_probability']*100:.1f}%)")

    # 5. Metrics & Samples
    print("\n[5/5] Testing GET /api/metrics and GET /api/samples...")
    m = client.metrics()
    acc = m["production_model"]["evaluation"]["accuracy"]
    s = client.samples()
    print(f"      [PASS] Accuracy: {acc*100:.2f}% | Samples Available: {len(s)}")

    print("\n" + "=" * 65)
    print("   ALL API ENDPOINT INTEGRATION TESTS PASSED (100%)")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
