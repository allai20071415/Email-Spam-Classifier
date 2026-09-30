# PROJECT REPORT
# Email Spam Classifier

## 1. Abstract
In the age of digital communication, spam emails pose a serious threat by flooding inboxes with irrelevant or malicious content. This project develops a Machine Learning-based Email Spam Classifier that automatically distinguishes spam from non-spam (ham) messages by analyzing their textual content.

The system uses Natural Language Processing (NLP), TF-IDF feature extraction, and supervised learning algorithms. Three algorithms—Naive Bayes, Logistic Regression, and Support Vector Machine (SVM)—are implemented and compared. The trained model can classify new messages in real time and can visualize important terms associated with spam and legitimate messages.

## 2. Introduction
Spam messages can reduce productivity and may contain unwanted promotions, scams, or malicious links. Manual filtering is inefficient when the volume of messages is high. Machine learning provides an automated approach by learning patterns from labeled examples.

This project follows the supplied specification: text processing extracts meaningful features, supervised learning identifies patterns, and the resulting model predicts whether new messages are spam or ham.

## 3. Problem Statement
To develop a lightweight Machine Learning system capable of automatically classifying email/message content into two categories:
- Spam
- Ham (legitimate)

## 4. Objectives
1. Collect and prepare labeled message data.
2. Apply NLP-based text preprocessing.
3. Convert text into numerical TF-IDF features.
4. Train multiple supervised ML models.
5. Compare model performance using standard metrics.
6. Save the best model for real-time inference.
7. Visualize important spam-indicating terms.
8. Provide a reusable and scalable prototype.

## 5. Scope
The prototype focuses on text classification. The supplied project specification also identifies possible future extension toward phishing emails, scams, and advertisements.

## 6. Features
- Spam/ham detection
- Text feature extraction
- Real-time classification
- Model comparison
- Evaluation metrics
- Keyword importance visualization
- Local, lightweight execution

## 7. Technologies Used
### Programming Language
Python

### Libraries
- scikit-learn — model training and evaluation
- pandas and NumPy — data handling
- NLTK — NLP preprocessing
- Matplotlib and Seaborn — visualization
- joblib — model persistence

### Algorithms
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

## 8. Dataset
The supplied project specification lists the SpamAssassin Dataset or UCI SMS Spam Collection Dataset, including the Kaggle SMS Spam Collection reference.

For immediate reproducibility, this repository contains a small demonstration dataset in `data/messages.csv`. It contains labeled `spam` and `ham` examples and is intended for software demonstration rather than a claim of production-level accuracy.

For a final deployment, a larger labeled dataset should be substituted and the model retrained.

## 9. System Workflow
```text
Input Message
      |
      v
Text Cleaning / NLP
      |
      v
TF-IDF Feature Extraction
      |
      v
ML Classifier
      |
      +--> Naive Bayes
      +--> Logistic Regression
      +--> SVM
      |
      v
Model Evaluation
      |
      v
Best Model Saved
      |
      v
New Message --> Spam / Ham
```

## 10. Methodology
### Step 1 — Data Loading
The labeled CSV is loaded using pandas.

### Step 2 — Preprocessing
Text is normalized to lowercase, URLs and email addresses are replaced with markers, punctuation is removed, stopwords are filtered, and words are lemmatized.

### Step 3 — Feature Extraction
TF-IDF converts text into numerical features. Unigrams and bigrams are used so that both individual terms and short phrases can contribute to classification.

### Step 4 — Model Training
The dataset is split into training and testing portions. Naive Bayes, Logistic Regression, and SVM models are trained.

### Step 5 — Evaluation
Accuracy, precision, recall, and F1-score are calculated. A confusion matrix is generated for the selected model.

### Step 6 — Model Selection
The implementation selects the model with the highest F1-score on the held-out test set. This is a project implementation rule, not a claim that the model is universally superior.

### Step 7 — Real-Time Prediction
The saved model accepts a new message and returns the predicted class.

## 11. Evaluation Metrics
- **Accuracy:** proportion of correctly classified messages.
- **Precision:** proportion of predicted spam messages that are actually spam.
- **Recall:** proportion of actual spam messages detected by the classifier.
- **F1-score:** harmonic mean of precision and recall.

## 12. Results
Run:

```bash
python -m src.train
```

The command generates:
- `reports/model_comparison.csv`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`
- `models/model_metrics.json`

Because the included dataset is a small demonstration dataset, the generated metrics should be used for demonstration/testing only. Production claims require evaluation on a substantially larger, representative dataset.


## 12A. Demonstration Dataset Results

| model               |   accuracy |   precision |   recall |   f1 |
|:--------------------|-----------:|------------:|---------:|-----:|
| Naive Bayes         |          1 |           1 |        1 |    1 |
| Logistic Regression |          1 |           1 |        1 |    1 |
| SVM                 |          1 |           1 |        1 |    1 |

These values are obtained from the included small demonstration dataset and are not production performance claims.
## 13. Real-Time Prediction
Run:

```bash
python -m src.predict
```

Example input:
```text
Congratulations! You have won a free prize. Click now!
```

Expected class:
```text
SPAM
```

The exact confidence depends on the trained model and dataset.

## 14. Keyword Visualization
Run:

```bash
python -m src.visualize
```

The system creates `reports/keyword_importance.png` when the selected classifier provides linear coefficients.

## 15. Advantages
- Automatic classification
- Fast local inference
- Uses established NLP and ML methods
- Multiple algorithms for comparison
- Easy to retrain
- Extendable to related message-security classification tasks

## 16. Limitations
- Demonstration dataset is small.
- Real-world email data can be more complex than short SMS-like messages.
- Sender behavior is not modeled directly in this prototype.
- Model quality depends on dataset quality and distribution.
- Confidence values are not available from every supported classifier.

## 17. Future Enhancements
1. Train on a larger SpamAssassin or UCI-derived dataset.
2. Add phishing URL and domain analysis.
3. Add attachment metadata analysis.
4. Add sender/domain reputation features.
5. Build a web dashboard.
6. Add continuous retraining and monitoring.
7. Add multilingual spam detection.
8. Integrate the classifier with an email service.

## 18. Conclusion
The project demonstrates how NLP and supervised machine learning can be combined to automatically classify messages as spam or legitimate. The implementation provides preprocessing, TF-IDF feature extraction, three supervised algorithms, model comparison, persistence, real-time prediction, and visualization.

The prototype can serve as a foundation for a larger spam-filtering system and can be extended toward phishing, scam, and advertisement detection as identified in the project specification.

## 19. References
- Project specification supplied for this project.
- scikit-learn documentation
- NLTK documentation
- pandas documentation
- Matplotlib documentation
- Seaborn documentation
