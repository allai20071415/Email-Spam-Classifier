import argparse
from pathlib import Path
import joblib
from .preprocess import clean_text
from .config import MODEL_DIR

def load_model():
    path = MODEL_DIR / "best_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "Model not found. Run: python -m src.train"
        )
    return joblib.load(path)

def predict_message(model, message):
    cleaned = clean_text(message)
    label = model.predict([cleaned])[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba([cleaned])[0]))
    return label, confidence

def main():
    parser = argparse.ArgumentParser(description="Email Spam Classifier")
    parser.add_argument("--text", help="Message/email to classify")
    args = parser.parse_args()

    model = load_model()
    message = args.text if args.text else input("Enter email/message: ")
    label, confidence = predict_message(model, message)

    print("\nPrediction:", "SPAM" if label == "spam" else "HAM (LEGITIMATE)")
    if confidence is not None:
        print(f"Confidence: {confidence * 100:.2f}%")
    print("")

if __name__ == "__main__":
    main()
