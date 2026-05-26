from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ml.config.config import RANDOM_STATE, CV_FOLDS
from ml.utils.logger import logger

def get_calibrated_svm():
    """
    Returns a LinearSVC wrapped in CalibratedClassifierCV so that it 
    provides predict_proba outputs for stacking.
    """
    base_svc = LinearSVC(random_state=RANDOM_STATE, max_iter=2000, dual=False)
    return CalibratedClassifierCV(estimator=base_svc, cv=3)

def build_text_stacking_ensemble() -> StackingClassifier:
    """
    Builds the Stacking Classifier for text scam detection.
    Base models: Logistic Regression, Multinomial NB, Calibrated SVM, Random Forest, LightGBM, CatBoost.
    Meta-learner: XGBoost.
    """
    logger.info("Assembling text stacking ensemble models...")
    
    base_estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")),
        ("nb", MultinomialNB()),
        ("svm", get_calibrated_svm()),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)),
        ("lgb", LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1, n_jobs=-1)),
        ("cb", CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, auto_class_weights="Balanced"))
    ]
    
    meta_learner = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        verbosity=0,
        n_jobs=-1
    )
    
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=CV_FOLDS,
        n_jobs=None
    )
    
    return stacking_model

def build_url_stacking_ensemble() -> StackingClassifier:
    """
    Builds the Stacking Classifier for URL phishing detection.
    Base models: Logistic Regression, Calibrated SVM, Random Forest, LightGBM, CatBoost.
    (Multinomial NB is excluded because features in phishing.csv contain negative values like -1).
    Meta-learner: XGBoost.
    """
    logger.info("Assembling URL stacking ensemble models...")
    
    base_estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")),
        ("svm", get_calibrated_svm()),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)),
        ("lgb", LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1, n_jobs=-1)),
        ("cb", CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, auto_class_weights="Balanced"))
    ]
    
    meta_learner = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        verbosity=0,
        n_jobs=-1
    )
    
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=CV_FOLDS,
        n_jobs=None
    )
    
    return stacking_model
