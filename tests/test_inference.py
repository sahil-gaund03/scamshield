import os
import sys
import pytest

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.pipelines.preprocessing import TextPreprocessor
from ml.pipelines.features import URLFeatureExtractor
from ml.pipelines.inference import ScamShieldInference

def test_text_preprocessor():
    preprocessor = TextPreprocessor(method="stemming")
    text = "Subject: Urgent notice! Click http://example.com now. Please contact us."
    cleaned = preprocessor.clean(text)
    
    # Assert lowercase
    assert cleaned == cleaned.lower()
    # Assert NLTK stemming ran and stopwords are removed
    assert "urgent" in cleaned or "urg" in cleaned
    assert "subject" not in cleaned

def test_url_detector():
    inference = ScamShieldInference()
    
    # Valid URLs
    assert inference.is_url("http://google.com") is True
    assert inference.is_url("https://verify.security-update.bank.co.uk/login") is True
    assert inference.is_url("www.malicious-domain.org/ref=123") is True
    
    # Non-URLs / standard text
    assert inference.is_url("Hey what are you doing tonight?") is False
    assert inference.is_url("Congratulations! You won a cash prize. Claim now.") is False

def test_url_feature_extractor():
    url = "https://www.paypal.com-security-alert.confirm-webapps.com/webscr?cmd=_login-run"
    features = URLFeatureExtractor.extract_features(url)
    
    # Check shape: 1 sample, 30 numerical features
    assert features.shape == (1, 30)
    # Check that it extracted features (like length, prefix_suffix, etc.)
    # The domain uses "-" so prefix_suffix should be 1 (indicating presence of "-")
    # Features are scaled/processed, let's verify type
    assert features.dtype == float

def test_inference_routing():
    inference = ScamShieldInference()
    
    # Check models are loaded
    assert inference.text_model is not None
    assert inference.url_model is not None
    assert inference.text_vectorizer is not None
    
    # Predict on a text message
    text_msg = "Urgent action required! Claim your free reward by clicking the link today."
    res_text = inference.predict(text_msg)
    assert res_text["input_type"] == "TEXT"
    assert "prediction" in res_text
    assert res_text["label"] in [0, 1]
    assert 0.0 <= res_text["confidence"] <= 1.0
    
    # Predict on a URL
    url = "http://verify-paypal-account-support.com"
    res_url = inference.predict(url)
    assert res_url["input_type"] == "URL"
    assert "prediction" in res_url
    assert res_url["label"] in [0, 1]
    assert 0.0 <= res_url["confidence"] <= 1.0
