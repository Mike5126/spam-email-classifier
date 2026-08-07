# Spam Email Classification

A machine learning pipeline that classifies emails as **spam** or **not spam** using TF-IDF feature extraction and classic ML models.

**Dataset:** [Spam Email Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset) — a combination of the TREC 2007 Public Spam Corpus and the Enron-Spam Dataset.

## Features

- Text cleaning (URL/HTML removal, punctuation stripping, normalization)
- TF-IDF vectorization (unigrams + bigrams, up to 20,000 features)
- Trains and compares 3 models: Naive Bayes, Logistic Regression, Random Forest
- Evaluation with Accuracy, Precision, Recall, F1, AUC, confusion matrices, and ROC curves
- Saves the best-performing model for inference on new emails

## Project Structure

```
.
├── spam_classification_notebook.ipynb    # Jupyter notebook version (with explanations)
├── combined_data.csv                     # Dataset (download from Kaggle, not included)
└── output/                                # Generated after running: plots, metrics, saved model
```

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage

1. Download `combined_data.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset) and place it in the project root.
2. Open `spam_classification_notebook.ipynb` in Jupyter/VS Code and run all cells.

3. Results are saved to `output/`:
   - `model_comparison.csv` — metrics for all models
   - `label_distribution.png`, `confusion_matrix_*.png`, `model_f1_comparison.png`, `roc_curves.png`
   - `best_model.joblib`, `tfidf_vectorizer.joblib` — the trained model + vectorizer

## Predicting on a New Email

The notebook defines a `predict_new_email()` function that loads the saved model and vectorizer:

```python
predict_new_email("Congratulations! You won a free iPhone, click here now!!!")
# -> Prediction: SPAM 🚫  (spam probability: 52.50%)
```
