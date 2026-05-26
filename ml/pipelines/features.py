import re
import math
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from ml.config.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE

# Suspicious keywords for text scoring
SCAM_KEYWORDS = [
    "congratulations", "urgent", "won", "claim", "prize", "free", "gift card", 
    "selected", "cash", "verify", "suspend", "compromise", "alert", "winner", 
    "account", "password", "bank", "update", "service", "expire", "limited"
]

class TextFeatureEngineer:
    def __init__(self, max_features: int = TFIDF_MAX_FEATURES, ngram_range: tuple = TFIDF_NGRAM_RANGE):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def extract_meta_features(self, raw_texts: list) -> np.ndarray:
        """
        Extracts structural and statistical metadata from raw texts.
        """
        meta_features = []
        for text in raw_texts:
            if not isinstance(text, str):
                text = ""
                
            length_char = len(text)
            length_word = len(text.split()) if length_char > 0 else 0
            
            # Capitalization ratio
            upper_ratio = sum(1 for c in text if c.isupper()) / length_char if length_char > 0 else 0
            
            # Digit frequency
            digit_ratio = sum(1 for c in text if c.isdigit()) / length_char if length_char > 0 else 0
            
            # Special symbol frequencies
            symbol_freq = sum(1 for c in text if c in "$%!@#^&*()_+-=[]{}|;':\",./<>?") / length_char if length_char > 0 else 0
            
            # Keyword matching score
            keyword_score = sum(text.lower().count(kw) for kw in SCAM_KEYWORDS)
            
            meta_features.append([
                length_char,
                length_word,
                upper_ratio,
                digit_ratio,
                symbol_freq,
                keyword_score
            ])
            
        return np.array(meta_features)

    def fit(self, cleaned_texts: pd.Series, raw_texts: list):
        """
        Fits the TF-IDF vectorizer and the standard scaler for metadata.
        """
        # Fit vectorizer
        self.vectorizer.fit(cleaned_texts)
        
        # Fit scaler
        meta_feats = self.extract_meta_features(raw_texts)
        self.scaler.fit(meta_feats)
        
        self.is_fitted = True
        return self

    def transform(self, cleaned_texts: pd.Series, raw_texts: list):
        """
        Transforms texts into a combined feature matrix (TF-IDF + Metadata).
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer is not fitted yet!")
            
        # TF-IDF sparse matrix
        tfidf_feats = self.vectorizer.transform(cleaned_texts)
        
        # Metadata dense matrix (then scaled)
        meta_feats = self.extract_meta_features(raw_texts)
        meta_scaled = self.scaler.transform(meta_feats)
        
        # Combine
        combined_feats = hstack([tfidf_feats, csr_matrix(meta_scaled)])
        return combined_feats

    def fit_transform(self, cleaned_texts: pd.Series, raw_texts: list):
        self.fit(cleaned_texts, raw_texts)
        return self.transform(cleaned_texts, raw_texts)


class URLFeatureExtractor:
    """
    Parses a raw URL string and extracts a 30-feature vector compatible 
    with the 'phishing.csv' dataset columns.
    Values: -1 (Legitimate/Safe), 0 (Suspicious/Neutral), 1 (Phishy/Scam)
    """
    @staticmethod
    def is_ip(domain: str) -> bool:
        # Regex to check if domain is an IPv4 address
        return bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain))

    @classmethod
    def extract_features(cls, url: str) -> np.ndarray:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
            
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        
        # 1. UsingIP
        using_ip = 1 if cls.is_ip(domain) else -1
        
        # 2. LongURL
        url_len = len(url)
        long_url = 1 if url_len >= 75 else (0 if url_len >= 54 else -1)
        
        # 3. ShortURL
        short_domains = ["bit.ly", "tinyurl.com", "t.co", "goog.gl", "is.gd", "cli.gs", "yfrog.com", "ow.ly"]
        short_url = 1 if any(sd in domain.lower() for sd in short_domains) else -1
        
        # 4. Symbol@
        symbol_at = 1 if "@" in url else -1
        
        # 5. Redirecting// (check if // is in path)
        redirect_double_slash = 1 if "//" in path else -1
        
        # 6. PrefixSuffix- (checks for '-' in domain)
        prefix_suffix = 1 if "-" in domain else -1
        
        # 7. SubDomains
        # Split domain by dots and check count (ignoring www)
        clean_domain = domain.replace("www.", "")
        parts = clean_domain.split(".")
        subdomains = 1 if len(parts) > 3 else (0 if len(parts) == 3 else -1)
        
        # 8. HTTPS
        https = -1 if parsed.scheme == "https" else 1
        
        # Heuristic rules or sensible defaults for website internal contents / external queries:
        domain_reg_len = -1
        favicon = -1
        non_std_port = -1
        https_domain_url = -1
        request_url = -1
        anchor_url = -1
        links_in_script = -1
        sfh = -1
        info_email = 1 if "mailto:" in url or "mail()" in url else -1
        abnormal_url = -1
        forwarding = -1
        status_bar = -1
        disable_right = -1
        popup = -1
        iframe = -1
        age_of_domain = -1
        dns_recording = -1
        traffic = -1
        page_rank = -1
        google_index = -1
        pointing_to_page = -1
        stats_report = -1
        
        feature_vector = [
            using_ip, long_url, short_url, symbol_at, redirect_double_slash,
            prefix_suffix, subdomains, https, domain_reg_len, favicon,
            non_std_port, https_domain_url, request_url, anchor_url,
            links_in_script, sfh, info_email, abnormal_url, forwarding,
            status_bar, disable_right, popup, iframe, age_of_domain,
            dns_recording, traffic, page_rank, google_index, pointing_to_page,
            stats_report
        ]
        
        return np.array(feature_vector).reshape(1, -1)
