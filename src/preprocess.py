import re

# A compact built-in English stopword set keeps the project offline-ready.
STOP_WORDS = {
    "a","an","and","are","as","at","be","been","but","by","for","from",
    "had","has","have","he","her","here","hers","him","his","how","i",
    "if","in","into","is","it","its","me","my","of","on","or","our","ours",
    "she","that","the","their","them","there","they","this","to","too",
    "was","we","were","what","when","where","which","who","why","will",
    "with","you","your","yours"
}

def _simple_lemma(token: str) -> str:
    """Small deterministic lemmatization fallback with no external corpus."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token

def clean_text(text: str) -> str:
    """Normalize text using lightweight NLP preprocessing."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        _simple_lemma(token)
        for token in text.split()
        if token not in STOP_WORDS and len(token) > 1
    ]
    return " ".join(tokens)
