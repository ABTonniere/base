import sys
import re
import string
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from scipy.sparse import hstack, csr_matrix

# Configuration
DATASET_PATH = "./spam.csv"
TEST_SIZE = 0.2
MODELS_DIR = Path("models")

def main():
    """Main training pipeline"""
    print("Starting spam detection pipeline...")
    
    # 1. Load and explore data
    print("\n=== Loading Data ===")
    df = pd.read_csv(DATASET_PATH, encoding="ISO-8859-1")
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
    # 2. Clean data
    print("\n=== Cleaning Data ===")
    df = df.drop_duplicates(subset=["message"])
    print(f"After removing duplicates: {len(df)} rows")
    
    # 3. Preprocess text and add features
    print("\n=== Preprocessing and Feature Engineering ===")
    df["message_clean"] = df["message"].str.lower()
    df["message_clean"] = df["message_clean"].apply(lambda x: re.sub(r"http\S+|www\S+|https\S+", "", x))
    df["message_clean"] = df["message_clean"].apply(lambda x: re.sub(r"\S+@\S+", "", x))
    df["message_clean"] = df["message_clean"].apply(lambda x: re.sub(r"\d+", "", x))
    df["message_clean"] = df["message_clean"].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
    df["message_clean"] = df["message_clean"].apply(lambda x: " ".join(x.split()))
    
    df["message_length"] = df["message"].apply(len)
    df["word_count"] = df["message"].apply(lambda x: len(x.split()))
    df["avg_word_length"] = df["message_length"] / df["word_count"]
    df["caps_count"] = df["message"].apply(lambda x: sum(1 for c in x if c.isupper()))
    df["caps_ratio"] = df["caps_count"] / df["message_length"]
    df["special_chars"] = df["message"].apply(lambda x: sum(1 for c in x if c in "!?$€£%"))
    
    # 4. Feature extraction
    print("\n=== Feature Extraction ===")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95, stop_words="english")
    X_tfidf = vectorizer.fit_transform(df["message_clean"])
    print(f"TF-IDF shape: {X_tfidf.shape}")
    
    numerical_features = ["message_length", "word_count", "avg_word_length", "caps_ratio", "special_chars"]
    X_numerical = df[numerical_features].values
    
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)
    X_numerical_sparse = csr_matrix(X_numerical_scaled)
    X_combined = hstack([X_tfidf, X_numerical_sparse])
    print(f"Combined features shape: {X_combined.shape}")
    
    # 5. Encode labels and split data
    print("\n=== Preparing Labels and Train-Test Split ===")
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    print(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
    
    # 6. Model training and hyperparameter tuning
    print("\n=== Training Model with Grid Search ===")
    param_grid = {
        "C": [0.1, 0.5, 1.0, 5.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # 7. Evaluation
    print("\n=== Model Evaluation ===")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 8. Save models
    print("\n=== Saving Models ===")
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("Models saved successfully!")

if __name__ == "__main__":
    main()
