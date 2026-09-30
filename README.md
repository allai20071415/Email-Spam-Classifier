# Email Spam Classifier

A complete Machine Learning project that classifies email/SMS text as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) and supervised learning.

## Project Overview
The classifier follows the project specification: it preprocesses message text, extracts text features, trains supervised ML models, evaluates their performance, predicts new messages in real time, and visualizes important spam-indicating words.

### Features
- Spam vs. ham classification
- NLP text preprocessing
- TF-IDF feature extraction
- Naive Bayes, Logistic Regression, and SVM models
- Model comparison and evaluation
- Real-time command-line prediction
- Important keyword visualization
- Lightweight local prototype
- Easily extensible toward phishing/scam/ad detection

## Technologies
- Python 3.10+
- pandas
- NumPy
- scikit-learn
- NLTK
- Matplotlib
- Seaborn
- joblib

## Dataset
The source specification allows the SpamAssassin Dataset or UCI SMS Spam Collection/Kaggle SMS Spam Collection. The repository includes a small **demo dataset** so the project runs immediately without downloading external data.

For a production-quality model, replace `data/messages.csv` with a larger labeled dataset using:
- `label`: `spam` or `ham`
- `text`: message/email content

The source document references the UCI SMS Spam Collection/Kaggle dataset. See the supplied project specification for the dataset reference.

## Installation

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Train the Models

```bash
python -m src.train
```

This creates:
- `models/best_model.joblib`
- `models/tfidf_vectorizer.joblib`
- `models/model_metrics.json`
- `reports/model_comparison.csv`
- `reports/confusion_matrix.png`
- `reports/keyword_importance.png`

## Run Real-Time Classification

```bash
python -m src.predict
```

Then enter a message when prompted.

Example:
```text
Enter email/message: Congratulations! You have won a free prize. Click now!
Prediction: SPAM
Confidence: 98.4%
```

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Project Structure

```text
Email_Spam_Classifier/
├── data/
│   └── messages.csv
├── models/
├── reports/
├── screenshots/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
├── tests/
│   └── test_classifier.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── PROJECT_REPORT.md
```

## Important Note
The included dataset is intentionally small and is suitable for demonstration/testing. Its accuracy should not be presented as production performance. For a strong final model, train on a substantially larger labeled dataset.

## GitHub

Create a repository named `email-spam-classifier`, then:

```bash
git init
git add .
git commit -m "Initial commit - Email Spam Classifier"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/email-spam-classifier.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

### Offline Support
The included preprocessing does not require downloading NLTK corpora, so the demo can run in restricted/offline environments.
