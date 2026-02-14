PhishNLP: Phishing Email Detection using NLP
A mini‑project for DATA 622 focused on detecting phishing and spam emails using Natural Language Processing (NLP) and machine‑learning techniques. The project walks through text preprocessing, feature extraction, model training, and evaluation.

Team
Rahul
Lalith

Objective
Build a classifier that labels email text as:

Label	Description
1	Spam / Phishing‑like
0	Legitimate (Ham)
 Dataset
File: emails.csv

Columns:

text — raw email content

label — ground‑truth classification (1 = spam/phish, 0 = ham)

 Methodology
1. Text Preprocessing
Lowercasing

Removing punctuation

Removing stopwords

Tokenization

Optional: Lemmatization/Stemming

2. Feature Engineering
TF‑IDF vectorization

Experimentation with n‑grams (unigrams, bigrams)

3. Model Training
Logistic Regression classifier

Train/test split

Optional hyperparameter tuning

4. Evaluation
Metrics include:

Confusion Matrix

Precision

Recall

F1‑Score

Accuracy

Results
The notebook outputs:

Confusion matrix

Classification metrics

Optional example predictions

Project Structure
Code
PhishNLP/
│
├── emails.csv
├── notebook.ipynb
├── README.md
└── requirements.txt
Running the Project
1. Install Dependencies
bash
pip install -r requirements.txt
2. Launch the Notebook
bash
jupyter notebook notebook.ipynb
3. Run All Cells
This will preprocess the data, train the model, and display evaluation metrics.

 Requirements
Example dependencies:

Code
pandas
numpy
scikit-learn
nltk
matplotlib
Future Enhancements
Try additional models (SVM, Random Forest, XGBoost)

Use embeddings (Word2Vec, GloVe, BERT)

Deploy as a web app or API

Add explainability tools (LIME, SHAP)
