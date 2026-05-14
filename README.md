![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Natural Language Processing Challenge – Fake News Classifier

## Introduction

This project applies Natural Language Processing techniques to classify news headlines as **fake (0)** or **real (1)**. It was developed as a group project for the Ironhack Data Science Bootcamp, combining classical ML models with a state-of-the-art Transformer model (ELECTRA).

---

## Dataset

The dataset was sourced from **Kaggle** and provided by Ironhack as part of this NLP challenge.

- **Source:** [Fake News Detection Datasets – Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- **Format:** Tab-separated (`.csv`), no header row, two columns: `label` and `text`
- **Labels:** `0` = Fake news, `1` = Real news, `2` = placeholder in test file (replaced by predictions)
- **Training rows:** 34,152 (raw) → 32,202 after deduplication and cleaning
- **Test rows:** 9,984

| File | Description |
|---|---|
| `dataset/training_data.csv` | Raw training set with labels 0/1 |
| `dataset/testing_data.csv` | Raw test set with placeholder label 2 |
| `dataset/train_clean.csv` | Cleaned training set (duplicates removed, `clean_text` column added) |
| `dataset/test_clean.csv` | Cleaned test set |

---

## Project Structure

```
project-3-nlp/
│
├── dataset/
│   ├── training_data.csv               # Raw training data
│   ├── testing_data.csv                # Raw test data
│   ├── train_clean.csv                 # Cleaned training data
│   └── test_clean.csv                  # Cleaned test data
│
├── models/
│   ├── JA_best_model_xgb.joblib        # Saved XGBoost model
│   └── JA_best_model_vectorizer_bow.joblib
│
├── finalNotebook_with_ELECTRA(1).ipynb # Main merged notebook (all models + ELECTRA)
├── main_Notebook_fake_news_classifier_2.ipynb  # Data preparation notebook
│
├── edits_Irish_model_fake_news_classifier_2.ipynb  # Irish – LR, SVM, Embeddings
├── Edited_GregoriNaive BayesExxtended.ipynb        # Gregori – Naive Bayes
├── edited_JA_RF_XGB_fake_news_.ipynb               # Juan Alvaro – RF, XGBoost
│
├── Irish_DistilBERT_fake_news.ipynb    # Irish – DistilBERT (Colab Pro)
│
├── presentationPowerPoint_updated_ELECTRA.pptx  # Final group presentation
├── outputs_predictions.csv             # Final predictions (Linear SVM)
└── README.md
```

---

## Notebooks

### Main Notebook – `finalNotebook_with_ELECTRA(1).ipynb`

The primary deliverable. Merges all individual model analyses into a single end-to-end pipeline and adds the ELECTRA Transformer model.

| Section | Description |
|---|---|
| 1 | Imports & Setup |
| 2 | Load Data |
| 3 | Exploratory Data Analysis |
| 4 | Text Preprocessing (deduplication, lowercasing, stopword removal, Porter stemming) |
| 5 | Train / Validation Split (80/20, stratified, `random_state=42`) |
| 6 | Vectorization Strategy (BoW and TF-IDF configurations per model family) |
| 7 | Train All Baseline Models (LR, SVM, NB, RF, XGBoost × BoW/TF-IDF) |
| 8 | Combined Model Comparison (bar chart, all team members) |
| 9 | Hyperparameter Tuning (GridSearchCV with Pipeline for LR, SVM, NB) |
| 10 | Final Best Model Selection (baseline + tuned combined ranking) |
| 11 | Detailed Evaluation of the Final Model (classification report, confusion matrix) |
| 12 | Cross-Validation of the Final Model (5-fold CV) |
| 13 | Feature Interpretation (top discriminating tokens via coefficients) |
| 14 | Error Analysis (misclassified headlines inspection) |
| 15 | Retrain on the Full Training Dataset |
| 16 | Generate Predictions for the Test File |
| 17 | Save the Final Prediction File |
| 18 | Final Conclusion |
| 19 | Additional Transformer Model: ELECTRA Small (results + training code) |
| 20 | Updated Final Conclusion After Adding ELECTRA |

### Individual Notebooks

> **Note:** Individual member notebooks are not included in this repository due to storage constraints. All analyses, models, and results from the individual notebooks have been fully merged into `finalNotebook_with_ELECTRA(1).ipynb`.

| Notebook | Member | Models covered |
|---|---|---|
| `edits_Irish_model_fake_news_classifier_2.ipynb` | Irish Levi | Logistic Regression, Linear SVM, Sentence Embeddings (Section 12) |
| `Edited_GregoriNaive BayesExxtended.ipynb` | Gregori | Multinomial Naive Bayes |
| `edited_JA_RF_XGB_fake_news_.ipynb` | Juan Alvaro | Random Forest, XGBoost |
|

---

## Models & Results

All models were evaluated on the same 20% validation split (6,441 rows, stratified).

| Model | Vectorizer | Validation Accuracy |
|---|---|---|
| **ELECTRA Small** *(Transformer)* | ELECTRA tokenizer | **0.9778** |
| Linear SVM | TF-IDF | 0.9362 |
| Linear SVM *(tuned)* | TF-IDF | 0.9359 |
| Logistic Regression | TF-IDF | 0.9340 |
| Logistic Regression *(tuned)* | TF-IDF | 0.9340 |
| Logistic Regression | Bag of Words | 0.9318 |
| Naive Bayes *(tuned)* | Bag of Words | 0.9306 |
| Naive Bayes | Bag of Words | 0.9306 |
| Naive Bayes | TF-IDF | 0.9278 |
| XGBoost | Bag of Words | 0.9244 |
| Linear SVM | Bag of Words | 0.9233 |
| XGBoost | TF-IDF | 0.9214 |
| Random Forest | TF-IDF | 0.9188 |
| Random Forest | Bag of Words | 0.9182 |

**Best classical model:** Linear SVM + TF-IDF (0.9362)  
**Best overall model:** ELECTRA Small Transformer (0.9778)

---

## Deliverables

1. **`finalNotebook_with_ELECTRA(1).ipynb`** — Complete end-to-end pipeline with all models, comparisons, and ELECTRA Transformer
2. **`outputs_predictions.csv`** — Final test predictions in the required format (tab-separated, label + text, no header)
3. **`outputs_predictions.electra.csv`** - Final prediction on the Electra Transformer
4. **`presentationPowerPoint_updated_ELECTRA.pptx`** — 10-minute group presentation covering methodology, results, and conclusions
5. **Accuracy estimation:** ~97.78% (ELECTRA) / ~93.6% (Linear SVM baseline)

---

## Setup

### Local (VS Code) — Classical models 

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# Install dependencies
pip install notebook scikit-learn pandas numpy matplotlib seaborn xgboost nltk sentence-transformers
```

### Google Colab Pro — ELECTRA Transformer

Open `finalNotebook_with_ELECTRA(1).ipynb` in Colab, set runtime to **GPU (A100)**, then run:

```python
!pip install transformers datasets -q
```

Set `RUN_ELECTRA_TRAINING = True` in Section 19 to retrain the model.

---

## Contributors

| Name | GitHub | Role |
|---|---|---|
| Irish Levi | [@irish16levi](https://github.com/irish16levi) | Logistic Regression · Linear SVM · Sentence Embeddings · DistilBERT |
| Gregori | [@Greg2828](https://github.com/Greg2828) | Naive Bayes |
| Juan Alvaro | [@juanalvaroromero-cell](https://github.com/juanalvaroromero-cell) | Random Forest · XGBoost · ELECTRA Transformer |

---

*Ironhack Data Science Bootcamp – NLP Project 3*
