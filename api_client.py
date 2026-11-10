"""
Python REST API Client for PhishGuard AI Backend.
Enables programmatic connection from Python scripts, bots, email hooks, and microservices.
"""

import urllib.request
import urllib.parse
import json
from typing import Dict, Any, List, Optional


class PhishGuardClient:
    """Client for connecting to the PhishGuard AI REST API."""

    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url.rstrip("/")

    def _request(self, endpoint: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        data = json.dumps(payload).encode("utf-8") if payload else None

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            try:
                return json.loads(err_body)
            except Exception:
                raise RuntimeError(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to backend at '{url}'. Reason: {e.reason}")

    def health(self) -> Dict[str, Any]:
        """Checks API health and model readiness."""
        return self._request("/api/health", method="GET")

    def predict(self, text: str, subject: str = "", sender: str = "") -> Dict[str, Any]:
        """Sends an email to the backend API for phishing threat analysis."""
        payload = {"text": text, "subject": subject, "sender": sender}
        return self._request("/api/predict", method="POST", payload=payload)

    def predict_batch(self, emails: List[str]) -> Dict[str, Any]:
        """Sends a list of emails to the backend API for batch analysis."""
        payload = {"emails": emails}
        return self._request("/api/batch", method="POST", payload=payload)

    def metrics(self) -> Dict[str, Any]:
        """Fetches trained model benchmark metrics."""
        return self._request("/api/metrics", method="GET")

    def samples(self) -> List[Dict[str, Any]]:
        """Fetches realistic preset test samples."""
        return self._request("/api/samples", method="GET")


if __name__ == "__main__":
    print("Testing PhishGuardClient connection to http://127.0.0.1:5000...")
    client = PhishGuardClient()
    try:
        health = client.health()
        print("Backend Status:", health)

        res = client.predict("URGENT: Verify your account immediately http://phish-secure.xyz")
        print("\nPrediction Test:")
        print("  Verdict:", res.get("prediction"))
        print("  Phishing Prob:", res.get("phishing_probability"))
        print("  Risk Level:", res.get("risk_level"))
        print("  Top Triggers:", [t["token"] for t in res.get("top_phishing_indicators", [])[:3]])
    except Exception as e:
        print("Connection failed:", e)
