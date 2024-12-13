import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import re
import numpy as np

# Load models and tokenizer
spam_model = load_model('spam_model.h5')
with open('phishing_model.pkl', 'rb') as f:
    phishing_model = pickle.load(f)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load hashing vectorizer
hashing_vec = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 4), n_features=5000)

# URL extraction utility
def extract_url(text):
    url_pattern = r'https?://[^\s]+'
    return re.findall(url_pattern, text)

# Function to predict label and convert spam to phishing only for phishing evaluation
def evaluate_model(text, is_phishing_eval=False):
    urls = extract_url(text)
    if urls:
        hashed_url = hashing_vec.transform(urls)
        phishing_score = phishing_model.predict_proba(hashed_url)[:, 1].mean()
        if phishing_score >= 0.5:
            return "Phishing"
    
    # If no URL, use spam model to classify the text
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    spam_score = spam_model.predict(padded, verbose=0)[0][0]
    
    return "Ham"

# Evaluation function for Spam Dataset
def evaluate_spam_dataset():
    print("Evaluating Spam Dataset...")
    spam_df = pd.read_csv('spam.csv', sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
    spam_df['label'] = spam_df['label'].map({'spam': 'Spam', 'ham': 'Ham'})
    
    y_true = spam_df['label']
    y_pred = spam_df['message'].apply(lambda x: evaluate_model(x, is_phishing_eval=False)) 
    
    print("\nSpam Dataset Results:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["Spam", "Ham"]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=["Spam", "Ham"]))

# Function to evaluate the Phishing Dataset with sampling
def evaluate_phishing_dataset():
    print("Evaluating Phishing Dataset (2% Sample)...")
    phishing_df = pd.read_csv('malicious_phish.csv').dropna()

    phishing_df['label'] = phishing_df['type'].map(lambda x: 'Phishing' if x != 'benign' else 'Ham')

    sampled_df = phishing_df.sample(frac=0.02, random_state=42)  # 2% sample, reproducible with random_state
    print(f"Using {len(sampled_df)} rows out of {len(phishing_df)} total rows.\n")

    y_true = sampled_df['label']
    y_pred = sampled_df['url'].apply(lambda x: evaluate_model(x, is_phishing_eval=True))  # Phishing Evaluation
    
    print(f"Number of true labels: {len(y_true)}")
    print(f"Number of predictions: {len(y_pred)}")
    
    # Print results
    print("\nPhishing Dataset Results:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["Phishing", "Ham"]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=["Phishing", "Ham"]))

# Run evaluations
if __name__ == "__main__":
    # evaluate_spam_dataset()
    # print("\n" + "="*50 + "\n")
    evaluate_phishing_dataset()
