"""
Data loading and preprocessing module for Email Phishing Detection.
Handles CSV ingestion, data validation, deduplication, and stratified train/test splitting.
"""

import os
import sys
import csv
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split

# Support large fields in email bodies
csv.field_size_limit(sys.maxsize)


def load_phishing_dataset(
    csv_path: str = "Email_phishing/phishing_email.csv",
    deduplicate: bool = False,
    min_length: int = 5
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """
    Loads email text and binary labels from CSV dataset.
    
    Args:
        csv_path: Path to the CSV file.
        deduplicate: Whether to remove duplicate text samples.
        min_length: Minimum text length in characters to retain.
        
    Returns:
        texts: List of cleaned email text strings.
        labels: List of integer labels (0 = Legitimate/Ham, 1 = Phishing).
        stats: Dictionary containing ingestion statistics.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found at: '{csv_path}'. "
            f"To retrain models from scratch, place your CSV dataset with 'text'/'text_combined' and 'label' columns at this path, "
            f"or use the pre-trained production model in 'models/phishing_detector.joblib'."
        )

    raw_count = 0
    empty_count = 0
    duplicate_count = 0
    seen_texts = set()

    texts: List[str] = []
    labels: List[int] = []

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # Detect column indices
        text_idx, label_idx = 0, 1
        if header:
            header_lower = [col.strip().lower() for col in header]
            if "text_combined" in header_lower and "label" in header_lower:
                text_idx = header_lower.index("text_combined")
                label_idx = header_lower.index("label")
            elif "text" in header_lower and "label" in header_lower:
                text_idx = header_lower.index("text")
                label_idx = header_lower.index("label")

        for row in reader:
            raw_count += 1
            if len(row) <= max(text_idx, label_idx):
                empty_count += 1
                continue

            raw_text = row[text_idx].strip()
            raw_label = row[label_idx].strip()

            if not raw_text or len(raw_text) < min_length:
                empty_count += 1
                continue

            try:
                label_val = int(raw_label)
                if label_val not in (0, 1):
                    continue
            except ValueError:
                continue

            if deduplicate:
                normalized = raw_text.lower()
                if normalized in seen_texts:
                    duplicate_count += 1
                    continue
                seen_texts.add(normalized)

            texts.append(raw_text)
            labels.append(label_val)

    phishing_count = sum(1 for l in labels if l == 1)
    legit_count = sum(1 for l in labels if l == 0)

    stats = {
        "raw_rows": raw_count,
        "valid_samples": len(texts),
        "skipped_empty_or_short": empty_count,
        "duplicates_removed": duplicate_count,
        "phishing_count": phishing_count,
        "legitimate_count": legit_count,
        "phishing_ratio": phishing_count / len(texts) if texts else 0.0,
    }

    return texts, labels, stats


def get_train_test_data(
    csv_path: str = "Email_phishing/phishing_email.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    deduplicate: bool = False
) -> Tuple[List[str], List[str], List[int], List[int], Dict[str, Any]]:
    """
    Loads dataset and performs stratified train/test split.
    
    Returns:
        X_train, X_test, y_train, y_test, stats
    """
    texts, labels, stats = load_phishing_dataset(csv_path=csv_path, deduplicate=deduplicate)
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    stats["train_size"] = len(X_train)
    stats["test_size"] = len(X_test)
    return X_train, X_test, y_train, y_test, stats


if __name__ == "__main__":
    print("Testing data loader...")
    _, _, stats = load_phishing_dataset()
    for k, v in stats.items():
        print(f"  {k}: {v}")
