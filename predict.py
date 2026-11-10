"""
Inference and Prediction CLI/Module for Email Phishing Detection.
Supports single email analysis, file-based inspection, interactive mode, and batch inference.
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Union

from src.model import PhishingDetector

DEFAULT_MODEL_PATH = "models/phishing_detector.joblib"
_CACHED_DETECTOR = None


def get_detector(model_path: str = DEFAULT_MODEL_PATH) -> PhishingDetector:
    """Loads and caches the trained phishing detector."""
    global _CACHED_DETECTOR
    if _CACHED_DETECTOR is None or getattr(_CACHED_DETECTOR, "_loaded_path", None) != model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                f"Please run 'python train.py' first to train the model."
            )
        _CACHED_DETECTOR = PhishingDetector.load(model_path)
        _CACHED_DETECTOR._loaded_path = model_path
    return _CACHED_DETECTOR


def analyze_email(
    email_text: str,
    model_path: str = DEFAULT_MODEL_PATH,
    top_k_clues: int = 6
) -> Dict[str, Any]:
    """
    Analyzes an email string and returns prediction, probability, risk level, and explanation.
    """
    detector = get_detector(model_path)
    clean_text = email_text.strip()
    if not clean_text:
        return {
            "prediction": "Unknown",
            "phishing_probability": 0.0,
            "legitimate_probability": 0.0,
            "risk_level": "EMPTY_INPUT",
            "message": "Empty or whitespace-only input text.",
            "top_phishing_indicators": [],
            "top_legitimate_indicators": [],
        }

    explanation = detector.explain(clean_text, top_k=top_k_clues)
    return explanation


def print_analysis(result: Dict[str, Any], text_snippet: str = ""):
    """Pretty-prints analysis results with risk assessment and keyword triggers."""
    is_phishing = result["prediction"] == "Phishing"
    p_phish = result["phishing_probability"] * 100
    p_legit = result["legitimate_probability"] * 100
    risk = result["risk_level"]

    border = "=" * 65
    print("\n" + border)
    if is_phishing:
        print(f" [!] SECURITY ALERT: PHISHING DETECTED  ({risk})")
    else:
        print(f" [✓] VERIFIED: LEGITIMATE EMAIL  ({risk})")
    print(border)

    if text_snippet:
        snippet = text_snippet.strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        print(f" Preview: {snippet}")
        print("-" * 65)

    print(f" Phishing Probability:   {p_phish:6.2f}%")
    print(f" Legitimate Probability: {p_legit:6.2f}%")
    print(f" Classification:         {result['prediction'].upper()}")
    print(f" Risk Assessment:        {risk}")

    phish_clues = result.get("top_phishing_indicators", [])
    if phish_clues:
        clue_str = ", ".join([f"'{c['token']}' (+{c['weight']:.2f})" for c in phish_clues])
        print(f" Top Phishing Triggers:  {clue_str}")

    legit_clues = result.get("top_legitimate_indicators", [])
    if legit_clues:
        legit_str = ", ".join([f"'{c['token']}' (-{c['weight']:.2f})" for c in legit_clues])
        print(f" Top Legitimate Tokens:  {legit_str}")
    print(border + "\n")


def interactive_mode(model_path: str = DEFAULT_MODEL_PATH):
    """Interactive CLI loop for testing arbitrary email bodies."""
    print("=" * 65)
    print("      INTERACTIVE PHISHING EMAIL DETECTOR")
    print("=" * 65)
    print("Type or paste email text below. Enter an empty line or type 'exit' to quit.\n")

    while True:
        try:
            line = input("Email Text > ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive detector.")
            break

        if not line.strip() or line.strip().lower() in ("exit", "quit", "q"):
            print("Exiting interactive detector.")
            break

        result = analyze_email(line, model_path=model_path)
        print_analysis(result, text_snippet=line)


def main():
    parser = argparse.ArgumentParser(description="Phishing Email Prediction and Analysis Tool")
    parser.add_argument("--text", "-t", type=str, help="Single email text string to analyze")
    parser.add_argument("--file", "-f", type=str, help="Path to text file containing an email")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model artifact")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive CLI prompt")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(model_path=args.model)
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found at '{args.file}'")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        res = analyze_email(content, model_path=args.model)
        print_analysis(res, text_snippet=content)
    elif args.text:
        res = analyze_email(args.text, model_path=args.model)
        print_analysis(res, text_snippet=args.text)
    else:
        # Default help or quick sample test
        parser.print_help()


if __name__ == "__main__":
    main()
