"""
Machine Learning Model Architecture for Email Phishing Detection.
Implements TF-IDF vectorization, probabilistic classification (SGDClassifier, LogisticRegression,
MultinomialNB, and Weighted Soft Voting Ensemble), along with local feature attribution (explanations).
"""

import os
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB


class PhishingDetector:
    """
    Production-ready Phishing Email Classifier.
    Combines TF-IDF n-gram vectorization with calibrated probabilistic classifiers.
    """

    def __init__(
        self,
        model_type: str = "ensemble",
        max_features: int = 75000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        random_state: int = 42,
    ):
        """
        Args:
            model_type: One of 'sgd', 'logistic', 'naive_bayes', 'ensemble'.
            max_features: Maximum vocabulary size for TF-IDF.
            ngram_range: (min_n, max_n) n-grams to extract.
            sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).
            random_state: Reproducibility seed.
        """
        self.model_type = model_type.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
            strip_accents="unicode",
        )

        self._init_models()
        self.is_fitted = False
        self.metadata: Dict[str, Any] = {}

    def _init_models(self):
        if self.model_type == "sgd":
            self.model = SGDClassifier(
                loss="modified_huber",
                penalty="l2",
                alpha=1e-5,
                max_iter=1000,
                random_state=self.random_state,
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                C=5.0,
                max_iter=1000,
                random_state=self.random_state,
            )
        elif self.model_type == "naive_bayes":
            self.model = MultinomialNB(alpha=0.1)
        elif self.model_type == "ensemble":
            self.model_sgd = SGDClassifier(
                loss="modified_huber",
                penalty="l2",
                alpha=1e-5,
                max_iter=1000,
                random_state=self.random_state,
            )
            self.model_lr = LogisticRegression(
                C=5.0,
                max_iter=1000,
                random_state=self.random_state,
            )
            self.model_nb = MultinomialNB(alpha=0.1)
            # Weights for soft-voting ensemble
            self.weights = (0.55, 0.35, 0.10)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Choose from 'sgd', 'logistic', 'naive_bayes', 'ensemble'.")

    def fit(self, texts: List[str], labels: List[int]) -> "PhishingDetector":
        """
        Fits vectorizer and classification model(s) on training data.
        """
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        if self.model_type == "ensemble":
            self.model_sgd.fit(X, y)
            self.model_lr.fit(X, y)
            self.model_nb.fit(X, y)
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        return self

    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Predicts class probabilities [P(Legitimate), P(Phishing)].
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        if isinstance(texts, str):
            texts = [texts]

        X = self.vectorizer.transform(texts)

        if self.model_type == "ensemble":
            p_sgd = self.model_sgd.predict_proba(X)
            p_lr = self.model_lr.predict_proba(X)
            p_nb = self.model_nb.predict_proba(X)
            w1, w2, w3 = self.weights
            return w1 * p_sgd + w2 * p_lr + w3 * p_nb
        else:
            return self.model.predict_proba(X)

    def predict(self, texts: Union[str, List[str]], threshold: float = 0.5) -> np.ndarray:
        """
        Predicts binary class (0 = Legitimate, 1 = Phishing) using decision threshold.
        """
        probs = self.predict_proba(texts)[:, 1]
        return (probs >= threshold).astype(int)

    def explain(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Explains prediction by computing token attribution scores.
        Identifies top words/phrases contributing to Phishing vs Legitimate decisions.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        X = self.vectorizer.transform([text])
        probs = self.predict_proba([text])[0]
        phishing_prob = float(probs[1])

        # Use linear model coefficients for attribution
        if self.model_type == "ensemble":
            coef = (
                self.weights[0] * self.model_sgd.coef_[0]
                + self.weights[1] * self.model_lr.coef_[0]
            )
        elif hasattr(self.model, "coef_"):
            coef = self.model.coef_[0]
        else:
            # For Naive Bayes, log prob difference
            coef = self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0]

        feature_names = self.vectorizer.get_feature_names_out()
        nz_indices = X.nonzero()[1]

        phishing_clues = []
        legit_clues = []

        for idx in nz_indices:
            token = feature_names[idx]
            tfidf_val = X[0, idx]
            impact = float(tfidf_val * coef[idx])
            if impact > 0:
                phishing_clues.append({"token": token, "weight": round(impact, 4)})
            elif impact < 0:
                legit_clues.append({"token": token, "weight": round(abs(impact), 4)})

        phishing_clues.sort(key=lambda x: x["weight"], reverse=True)
        legit_clues.sort(key=lambda x: x["weight"], reverse=True)

        # Risk classification
        if phishing_prob >= 0.85:
            risk_level = "CRITICAL_PHISHING"
        elif phishing_prob >= 0.50:
            risk_level = "HIGH_PHISHING"
        elif phishing_prob >= 0.25:
            risk_level = "SUSPICIOUS_MEDIUM"
        else:
            risk_level = "SAFE_LEGITIMATE"

        return {
            "prediction": "Phishing" if phishing_prob >= 0.5 else "Legitimate",
            "phishing_probability": round(phishing_prob, 4),
            "legitimate_probability": round(1.0 - phishing_prob, 4),
            "risk_level": risk_level,
            "top_phishing_indicators": phishing_clues[:top_k],
            "top_legitimate_indicators": legit_clues[:top_k],
        }

    def save(self, filepath: str):
        """
        Serializes entire model, vectorizer, and metadata to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath, compress=3)
        print(f"Model successfully saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "PhishingDetector":
        """
        Loads a saved model from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found at: {filepath}")
        loaded = joblib.load(filepath)
        return loaded
