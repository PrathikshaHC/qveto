"""
Email Phishing Detection Package.
"""
from src.data_loader import load_phishing_dataset, get_train_test_data
from src.model import PhishingDetector

__all__ = ["load_phishing_dataset", "get_train_test_data", "PhishingDetector"]
