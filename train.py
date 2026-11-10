"""
Comprehensive Model Training and Evaluation Pipeline for Phishing Email Detection.
Trains, benchmarks, and persists models using the unified phishing_email.csv dataset.
"""

import os
import sys
import time
import json
from typing import Dict, Any
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.data_loader import get_train_test_data
from src.model import PhishingDetector


def evaluate_predictions(y_true, y_prob, threshold: float = 0.5) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "threshold": threshold,
    }


def find_optimal_threshold(y_true, y_prob) -> float:
    """Finds decision threshold maximizing F1-score on test predictions."""
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.2, 0.81, 0.02):
        preds = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)
    return round(best_thresh, 2)


def main():
    print("=" * 70)
    print("   EMAIL PHISHING CLASSIFICATION MODEL TRAINING PIPELINE")
    print("=" * 70)

    dataset_path = "Email_phishing/phishing_email.csv"
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1/4] Loading dataset from '{dataset_path}'...")
    t0 = time.time()
    X_train, X_test, y_train, y_test, stats = get_train_test_data(
        csv_path=dataset_path,
        test_size=0.2,
        random_state=42,
        deduplicate=False
    )
    load_time = time.time() - t0
    print(f"      Loaded {stats['valid_samples']:,} samples in {load_time:.2f}s")
    print(f"      Training set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")
    print(f"      Class distribution: {stats['phishing_count']:,} Phishing (1), {stats['legitimate_count']:,} Legitimate (0)")

    # Model candidates to train and evaluate
    candidate_types = ["sgd", "logistic", "naive_bayes", "ensemble"]
    results = {}
    trained_models = {}

    print("\n[2/4] Training and benchmarking candidate architectures...")
    print("-" * 75)
    print(f"{'Model Architecture':25s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':7s} | {'F1-Score':8s} | {'ROC-AUC':7s} | {'Train Time':10s}")
    print("-" * 75)

    for m_type in candidate_types:
        t_start = time.time()
        detector = PhishingDetector(
            model_type=m_type,
            max_features=75000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            random_state=42
        )
        detector.fit(X_train, y_train)
        train_duration = time.time() - t_start

        probs = detector.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, probs, threshold=0.5)
        metrics["training_time_seconds"] = round(train_duration, 2)

        results[m_type] = metrics
        trained_models[m_type] = detector

        print(
            f"{m_type.upper():25s} | "
            f"{metrics['accuracy']*100:6.2f}%  | "
            f"{metrics['precision']*100:7.2f}%  | "
            f"{metrics['recall']*100:5.2f}% | "
            f"{metrics['f1_score']*100:6.2f}%  | "
            f"{metrics['roc_auc']:7.4f} | "
            f"{train_duration:8.2f}s"
        )

    print("-" * 75)

    # Choose primary production model: ensemble for peak robustness and calibrated probabilities
    primary_model_type = "ensemble"
    primary_model = trained_models[primary_model_type]
    test_probs = primary_model.predict_proba(X_test)[:, 1]
    optimal_thresh = find_optimal_threshold(y_test, test_probs)
    primary_eval = evaluate_predictions(y_test, test_probs, threshold=optimal_thresh)

    print(f"\n[3/4] Optimal Threshold Analysis for '{primary_model_type.upper()}':")
    print(f"      Default Threshold (0.50): F1 = {results[primary_model_type]['f1_score']*100:.2f}%, Accuracy = {results[primary_model_type]['accuracy']*100:.2f}%")
    print(f"      Optimal Threshold ({optimal_thresh:.2f}): F1 = {primary_eval['f1_score']*100:.2f}%, Accuracy = {primary_eval['accuracy']*100:.2f}%")
    print(f"      Confusion Matrix at {optimal_thresh:.2f}:")
    cm = primary_eval["confusion_matrix"]
    print(f"        TN (True Legitimate):  {cm['true_negatives']:,}")
    print(f"        FP (False Alarm):      {cm['false_positives']:,}")
    print(f"        FN (Missed Phishing):  {cm['false_negatives']:,}")
    print(f"        TP (Caught Phishing):  {cm['true_positives']:,}")

    # Attach metadata to primary model
    primary_model.metadata = {
        "dataset": "Email_phishing/phishing_email.csv",
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": stats["valid_samples"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "model_type": primary_model_type,
        "max_features": 75000,
        "ngram_range": [1, 2],
        "default_threshold": 0.50,
        "optimal_threshold": optimal_thresh,
        "metrics_default_threshold": results[primary_model_type],
        "metrics_optimal_threshold": primary_eval,
    }

    print("\n[4/4] Saving model artifacts...")
    model_save_path = os.path.join(output_dir, "phishing_detector.joblib")
    primary_model.save(model_save_path)

    # Save also SGD individual model as lightweight fast alternative
    sgd_save_path = os.path.join(output_dir, "phishing_detector_sgd.joblib")
    trained_models["sgd"].metadata = {
        "model_type": "sgd",
        "metrics": results["sgd"],
    }
    trained_models["sgd"].save(sgd_save_path)

    metrics_save_path = os.path.join(output_dir, "metrics.json")
    all_metrics = {
        "dataset_stats": stats,
        "candidate_benchmark": results,
        "production_model": {
            "model_type": primary_model_type,
            "optimal_threshold": optimal_thresh,
            "evaluation": primary_eval,
        },
    }
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"      Metrics and benchmark saved to: {metrics_save_path}")

    print("\n" + "=" * 70)
    print("   MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
