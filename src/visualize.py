import joblib
import matplotlib.pyplot as plt
from .config import MODEL_DIR, REPORT_DIR

def visualize_keywords():
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "coef_"):
        print("Keyword importance visualization is available for the selected linear model.")
        return

    names = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]
    pairs = sorted(zip(names, weights), key=lambda x: x[1])

    ham_words = pairs[:15]
    spam_words = pairs[-15:]
    words = [w for w, _ in ham_words + spam_words]
    values = [v for _, v in ham_words + spam_words]

    plt.figure(figsize=(10, 7))
    plt.barh(words, values)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Model coefficient")
    plt.title("Important Ham/Spam-Indicating Terms")
    plt.tight_layout()
    REPORT_DIR.mkdir(exist_ok=True)
    plt.savefig(REPORT_DIR / "keyword_importance.png", dpi=160)
    plt.close()
    print("Saved:", REPORT_DIR / "keyword_importance.png")

if __name__ == "__main__":
    visualize_keywords()
