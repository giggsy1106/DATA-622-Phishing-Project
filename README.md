PhishNLP: Phishing Email Detection using NLP

A mini-project for DATA 622 (Natural Language Processing) at University of Maryland, Baltimore County (UMBC).
This project focuses on detecting phishing and spam emails using Natural Language Processing (NLP) and machine learning techniques.

Team

Rahul Reddy Kota

Lalith

Problem Statement

Phishing emails are a major cybersecurity threat, often designed to trick users into revealing sensitive information.
This project aims to build a binary classification model that can automatically distinguish between:

Label	Description
1	Spam / Phishing Email
0	Legitimate Email (Ham)
Dataset

File: emails.csv

Columns:

text → Raw email content

label → Ground truth classification (1 = spam/phish, 0 = ham)

Methodology
1. Text Preprocessing

To prepare raw email data for modeling, we apply:

Lowercasing text

Removing punctuation and special characters

Removing stopwords (common words like "the", "is")

Tokenization

Optional lemmatization/stemming

2. Feature Engineering

We convert text into numerical form using:

TF-IDF Vectorization

N-grams (unigrams and bigrams)

This helps capture both individual words and short phrases.

3. Model Training

We use a Logistic Regression classifier as the baseline model:

Train-test split (e.g., 80/20)

Model training on processed features

Optional hyperparameter tuning

4. Evaluation Metrics

The model is evaluated using:

Confusion Matrix

Accuracy

Precision

Recall

F1-Score

These metrics help measure both correctness and the ability to detect phishing emails effectively.

Results

The notebook outputs:

Confusion Matrix visualization

Classification report (Precision, Recall, F1-score)

Sample predictions

The model demonstrates the effectiveness of TF-IDF + Logistic Regression as a baseline for phishing detection.

Project Structure
PhishNLP/
│
├── emails.csv            # Dataset
├── notebook.ipynb        # Main analysis and model training
├── README.md             # Project documentation
└── requirements.txt      # Dependencies
Installation
1. Clone the Repository
git clone https://github.com/your-username/PhishNLP.git
cd PhishNLP
2. Install Dependencies
pip install -r requirements.txt
Usage
Launch Jupyter Notebook
jupyter notebook notebook.ipynb
Run All Cells

This will:

Load the dataset

Preprocess text

Extract features

Train the model

Evaluate performance

Requirements

Example dependencies:

pandas
numpy
scikit-learn
nltk
matplotlib
Future Enhancements

Compare additional models (SVM, Random Forest, XGBoost, LightGBM)

Add character n-grams to detect obfuscated phishing patterns

Incorporate handcrafted features (URLs, keywords, punctuation patterns)

Handle class imbalance using SMOTE or class weights

Perform hyperparameter tuning using GridSearchCV

Improve explainability using feature importance or SHAP

Deploy the model as a REST API or web application

Key Learnings

Importance of text preprocessing in NLP pipelines

Effectiveness of TF-IDF for traditional ML models

Trade-offs between precision and recall in phishing detection

End-to-end workflow from data → model → evaluation
