import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import DATA_PATH, MODEL_DIR, REPORT_DIR, RANDOM_STATE, TEST_SIZE
from .preprocess import clean_text

def load_data():
    df = pd.read_csv(DATA_PATH)
    required = {"label", "text"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain 'label' and 'text' columns.")
    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["spam", "ham"])]
    df["clean_text"] = df["text"].apply(clean_text)
    return df

def build_models():
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=20000
    )
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", vectorizer),
            ("classifier", MultinomialNB())
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ]),
        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000)),
            ("classifier", LinearSVC(random_state=RANDOM_STATE))
        ]),
    }

def evaluate():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    results = []
    fitted = {}

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, pos_label="spam", zero_division=0),
            "recall": recall_score(y_test, pred, pos_label="spam", zero_division=0),
            "f1": f1_score(y_test, pred, pos_label="spam", zero_division=0),
        }
        results.append(metrics)
        fitted[name] = model

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    # Confusion matrix for best model
    best_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_pred, labels=["ham", "spam"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # Save metrics and a text report
    report = classification_report(y_test, best_pred, labels=["ham", "spam"], zero_division=0)
    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    metadata = {
        "best_model": best_name,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": results_df.to_dict(orient="records"),
    }
    (MODEL_DIR / "model_metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved to: {MODEL_DIR / 'best_model.joblib'}")
    print("\nClassification report:\n", report)

if __name__ == "__main__":
    evaluate()
