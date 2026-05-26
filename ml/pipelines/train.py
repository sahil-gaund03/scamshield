import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, roc_curve, confusion_matrix,
    classification_report
)

# Dynamically resolve project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.config import config
from ml.utils.logger import logger
from ml.pipelines.preprocessing import TextPreprocessor
from ml.pipelines.features import TextFeatureEngineer
from ml.pipelines.model import build_text_stacking_ensemble, build_url_stacking_ensemble

def save_evaluation_plots(y_true, y_pred_prob, prefix: str):
    """
    Saves ROC, Precision-Recall, and Confusion Matrix plots for reporting.
    """
    reports_dir = config.BASE_DIR / "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. ROC & PR Curves
    plt.figure(figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix.upper()} ROC Curve")
    plt.legend(loc="lower right")
    
    # PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(rec, prec)
    
    plt.subplot(1, 2, 2)
    plt.plot(rec, prec, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"{prefix.upper()} Precision-Recall Curve")
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plot_path = reports_dir / f"{prefix}_evaluation_curves.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved evaluation curves plot to {plot_path}")

def train_text_model():
    """
    Loads text dataset, cleans, vectorizes, trains text stacking ensemble, 
    evaluates, and exports model and vectorizer.
    """
    logger.info("--- Starting Text Scam Model Training ---")
    
    if not os.path.exists(config.SMS_SPAM_RAW):
        raise FileNotFoundError(f"SMS scam dataset not found at {config.SMS_SPAM_RAW}")
        
    # Load dataset
    logger.info(f"Loading text dataset from {config.SMS_SPAM_RAW}")
    df = pd.read_csv(config.SMS_SPAM_RAW, encoding="latin-1")
    
    # Process columns
    df = df[['target', 'text']]
    df.columns = ['label', 'message']
    
    # Handle NaN
    df.dropna(subset=['message', 'label'], inplace=True)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Remove rows where mapping failed
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    
    # Preprocess text
    logger.info("Cleaning raw text data...")
    preprocessor = TextPreprocessor(method="stemming")
    df['clean_message'] = df['message'].apply(preprocessor.clean)
    
    # Ensure no empty cleaned strings
    df = df[df['clean_message'].str.strip() != ""]
    
    X_raw = df['message'].tolist()
    X_clean = df['clean_message']
    y = df['label'].values
    
    # Split
    logger.info("Splitting text dataset (stratified)...")
    X_train_raw, X_test_raw, X_train_clean, X_test_clean, y_train, y_test = train_test_split(
        X_raw, X_clean, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Feature Engineering
    logger.info("Fitting feature engineer (TF-IDF + Metadata features)...")
    feat_engineer = TextFeatureEngineer()
    X_train_feats = feat_engineer.fit_transform(X_train_clean, X_train_raw)
    X_test_feats = feat_engineer.transform(X_test_clean, X_test_raw)
    
    # Build & Fit model
    model = build_text_stacking_ensemble()
    logger.info("Fitting text Stacking Classifier model (this may take a minute)...")
    model.fit(X_train_feats, y_train)
    
    # Predict
    y_pred = model.predict(X_test_feats)
    y_pred_prob = model.predict_proba(X_test_feats)[:, 1]
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    logger.info(f"Text Model Evaluation Results:")
    logger.info(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    # Save plots
    save_evaluation_plots(y_test, y_pred_prob, "text_model")
    
    # Serialize
    logger.info(f"Saving text vectorizer to {config.TEXT_VECTORIZER_PATH}")
    joblib.dump(feat_engineer, config.TEXT_VECTORIZER_PATH)
    
    logger.info(f"Saving text model to {config.TEXT_MODEL_PATH}")
    joblib.dump(model, config.TEXT_MODEL_PATH)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc
    }

def train_url_model():
    """
    Loads URL dataset, trains URL stacking ensemble, evaluates, and exports model.
    """
    logger.info("--- Starting URL Phishing Model Training ---")
    
    if not os.path.exists(config.PHISHING_RAW):
        raise FileNotFoundError(f"Phishing dataset not found at {config.PHISHING_RAW}")
        
    # Load dataset
    logger.info(f"Loading URL dataset from {config.PHISHING_RAW}")
    df = pd.read_csv(config.PHISHING_RAW)
    
    # Separate features and labels
    X = df.drop(columns=['class'])
    # Map -1 (phishing) -> 1, 1 (legitimate) -> 0
    y = df['class'].map({-1: 1, 1: 0}).values
    
    # Split
    logger.info("Splitting URL dataset (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Build & Fit model
    model = build_url_stacking_ensemble()
    logger.info("Fitting URL Stacking Classifier model...")
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    logger.info(f"URL Model Evaluation Results:")
    logger.info(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    # Save plots
    save_evaluation_plots(y_test, y_pred_prob, "url_model")
    
    # Serialize
    logger.info(f"Saving URL model to {config.URL_MODEL_PATH}")
    joblib.dump(model, config.URL_MODEL_PATH)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc
    }

if __name__ == "__main__":
    logger.info("Starting ScamShield AI System Training Pipeline...")
    
    metrics = {}
    try:
        text_metrics = train_text_model()
        metrics["text_model"] = text_metrics
    except Exception as e:
        logger.error(f"Error training text model: {e}", exc_info=True)
        
    try:
        url_metrics = train_url_model()
        metrics["url_model"] = url_metrics
    except Exception as e:
        logger.error(f"Error training URL model: {e}", exc_info=True)
        
    # Write metrics to reports folder
    reports_dir = config.BASE_DIR / "reports"
    os.makedirs(reports_dir, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"All training completed. Evaluation metrics saved to {metrics_path}")
