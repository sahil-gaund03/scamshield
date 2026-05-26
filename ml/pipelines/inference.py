import os
import re
import joblib
import numpy as np
from ml.config import config
from ml.utils.logger import logger
from ml.pipelines.preprocessing import TextPreprocessor
from ml.pipelines.features import URLFeatureExtractor

# URL detection pattern
URL_REGEX = re.compile(
    r'^(?:http|ftp)s?://' # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
    r'localhost|' # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
    r'(?::\d+)?' # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE
)

class ScamShieldInference:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ScamShieldInference, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        logger.info("Initializing ScamShield Inference Engine...")
        self.preprocessor = TextPreprocessor(method="stemming")
        self.text_vectorizer = None
        self.text_model = None
        self.url_model = None
        
        self.load_models()
        self._initialized = True

    def load_models(self):
        """
        Loads the serialized text model, text vectorizer, and URL model from disk.
        """
        try:
            if os.path.exists(config.TEXT_VECTORIZER_PATH):
                logger.info(f"Loading text vectorizer from {config.TEXT_VECTORIZER_PATH}")
                self.text_vectorizer = joblib.load(config.TEXT_VECTORIZER_PATH)
            else:
                logger.warning(f"Text vectorizer not found at {config.TEXT_VECTORIZER_PATH}")

            if os.path.exists(config.TEXT_MODEL_PATH):
                logger.info(f"Loading text model from {config.TEXT_MODEL_PATH}")
                self.text_model = joblib.load(config.TEXT_MODEL_PATH)
            else:
                logger.warning(f"Text model not found at {config.TEXT_MODEL_PATH}")

            if os.path.exists(config.URL_MODEL_PATH):
                logger.info(f"Loading URL model from {config.URL_MODEL_PATH}")
                self.url_model = joblib.load(config.URL_MODEL_PATH)
            else:
                logger.warning(f"URL model not found at {config.URL_MODEL_PATH}")
                
        except Exception as e:
            logger.error(f"Error loading model files: {e}", exc_info=True)

    def is_url(self, text: str) -> bool:
        """
        Heuristic check to identify if a string is a URL.
        """
        text = text.strip()
        # Direct starting checks
        if text.lower().startswith(("http://", "https://", "www.")):
            return True
        # Regex check
        return bool(URL_REGEX.match(text))

    def predict(self, text: str) -> dict:
        """
        Predicts if the input string is a scam/phishing attempt.
        Detects type, preprocesses, vectorizes, and runs classification.
        """
        text_clean = text.strip()
        if not text_clean:
            return {
                "input_type": "UNKNOWN",
                "prediction": "SAFE",
                "label": 0,
                "confidence": 0.0,
                "details": "Empty input"
            }

        # 1. Route Input (URL vs Text)
        if self.is_url(text_clean):
            return self._predict_url(text_clean)
        else:
            return self._predict_text(text_clean)

    def _predict_url(self, url: str) -> dict:
        """
        Runs URL feature extraction and stacking classification.
        """
        if self.url_model is None:
            return {
                "input_type": "URL",
                "prediction": "ERROR",
                "label": -1,
                "confidence": 0.0,
                "details": "URL Model not loaded"
            }
            
        try:
            # Feature extraction
            feats = URLFeatureExtractor.extract_features(url)
            
            # Predict
            pred = int(self.url_model.predict(feats)[0])
            prob = float(self.url_model.predict_proba(feats)[0][pred])
            
            # Since phishing.csv mapping maps phishy to 1 and legimate to 0
            prediction_label = "PHISHING" if pred == 1 else "SAFE"
            
            return {
                "input_type": "URL",
                "prediction": prediction_label,
                "label": pred,
                "confidence": prob,
                "details": f"Processed as URL. Features matched: {feats.tolist()[0][:8]}"
            }
        except Exception as e:
            logger.error(f"Failed to run URL prediction: {e}", exc_info=True)
            return {
                "input_type": "URL",
                "prediction": "ERROR",
                "label": -1,
                "confidence": 0.0,
                "details": f"URL Prediction failed: {str(e)}"
            }

    def _predict_text(self, text: str) -> dict:
        """
        Runs NLP text preprocessing, feature engineering, and stacking classification.
        """
        if self.text_model is None or self.text_vectorizer is None:
            return {
                "input_type": "TEXT",
                "prediction": "ERROR",
                "label": -1,
                "confidence": 0.0,
                "details": "Text model or vectorizer not loaded"
            }
            
        try:
            # NLP clean
            cleaned_text = self.preprocessor.clean(text)
            
            # Vectorize
            feats = self.text_vectorizer.transform([cleaned_text], [text])
            
            # Predict
            pred = int(self.text_model.predict(feats)[0])
            prob = float(self.text_model.predict_proba(feats)[0][pred])
            
            prediction_label = "SCAM" if pred == 1 else "SAFE"
            
            return {
                "input_type": "TEXT",
                "prediction": prediction_label,
                "label": pred,
                "confidence": prob,
                "details": f"Cleaned tokens: '{cleaned_text}'"
            }
        except Exception as e:
            logger.error(f"Failed to run text prediction: {e}", exc_info=True)
            return {
                "input_type": "TEXT",
                "prediction": "ERROR",
                "label": -1,
                "confidence": 0.0,
                "details": f"Text Prediction failed: {str(e)}"
            }
