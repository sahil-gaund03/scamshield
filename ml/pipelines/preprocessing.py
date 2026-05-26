import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from ml.utils.logger import logger

# Initialize NLTK components, download if missing
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

# Text cleaning components
try:
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = set()

# Basic slang mapping dictionary for normalization
SLANG_MAP = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "txt": "text",
    "msg": "message",
    "pls": "please",
    "plz": "please",
    "gr8": "great",
    "b4": "before",
    "c": "see",
    "w8": "wait",
    "l8r": "later",
    "im": "i am",
    "idk": "i do not know",
    "omw": "on my way",
    "fyi": "for your information"
}

class TextPreprocessor:
    def __init__(self, method: str = "stemming"):
        """
        Initializes the text preprocessor.
        method: 'stemming' (PorterStemmer) or 'lemmatization' (WordNetLemmatizer)
        """
        self.method = method.lower()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def normalize_slang(self, text: str) -> str:
        """
        Replaces short hands / internet slang with full words.
        """
        words = text.split()
        normalized_words = [SLANG_MAP.get(word, word) for word in words]
        return " ".join(normalized_words)

    def clean(self, text: str) -> str:
        """
        Cleans the input text by applying standard NLP cleaning steps.
        """
        if not isinstance(text, str):
            return ""

        # 1. Lowercasing
        text = text.lower()

        # 2. Strip HTML tags
        text = re.sub(r"<.*?>", " ", text)

        # 3. Strip URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # 4. Normalize slang/short hands
        text = self.normalize_slang(text)

        # 5. Keep only alphabetic characters
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # 6. Tokenize & remove stopwords / short tokens
        words = text.split()
        cleaned_words = [word for word in words if word not in STOPWORDS and len(word) > 1]

        # 7. Apply Stemming or Lemmatization
        if self.method == "lemmatization":
            processed_words = [self.lemmatizer.lemmatize(word) for word in cleaned_words]
        else:
            processed_words = [self.stemmer.stem(word) for word in cleaned_words]

        return " ".join(processed_words)
