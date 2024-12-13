import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import re

# Load and preprocess SMS spam data
df = pd.read_csv('spam.csv', sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Split data for spam detection
X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42)

# Tokenize and pad sequences for spam RNN model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_spam)
X_train_spam_seq = tokenizer.texts_to_sequences(X_train_spam)
X_test_spam_seq = tokenizer.texts_to_sequences(X_test_spam)
X_train_spam_pad = pad_sequences(X_train_spam_seq, maxlen=100)
X_test_spam_pad = pad_sequences(X_test_spam_seq, maxlen=100)

# Define the spam RNN model
def build_rnn_model():
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=100),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the spam detection model
spam_model = build_rnn_model()
spam_model.fit(X_train_spam_pad, y_train_spam, epochs=5, batch_size=32)
spam_model.save('spam_model.h5')

# Load phishing URL dataset
url_data = pd.read_csv('malicious_phish.csv').dropna()

# Prepare data
X_url = url_data['url']
y_url = url_data['type'].map({'benign': 0, 'phishing': 1, 'malware': 1, 'defacement': 1})  # Map labels
url_data = url_data.dropna(subset=['type'])  # Remove invalid entries

# Vectorize URLs using HashingVectorizer
hashing_vec = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 4), n_features=5000)
X_url_hashed = hashing_vec.fit_transform(X_url)

# Train-test split for URL data
X_train_url, X_test_url, y_train_url, y_test_url = train_test_split(
    X_url_hashed, y_url, test_size=0.3, random_state=42)

# Train the SGDClassifier
phishing_model = SGDClassifier(loss='log_loss', max_iter=800, tol=1e-3, random_state=42)
phishing_model.fit(X_train_url, y_train_url)

# Save the phishing classifier
with open('phishing_model.pkl', 'wb') as f:
    pickle.dump(phishing_model, f)

# Utility: Check if text contains a URL
def is_url_present(text):
    return bool(re.search(r'http[s]?://|www\.|\.com|\.org|\.net|\.co|\.edu|\.gov|\.info|\.tv|\.me|\.wiki|\.in|\.eu|\.asia|\.xyz|\.ly|\.site', text))

# Function to extract URL from the text
def extract_url(text):
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    return urls

# Predict using the individual models
def predict_text(text):
    if is_url_present(text):
        hashed_url = hashing_vec.transform([text])
        phishing_score = phishing_model.predict_proba(hashed_url)[:, 1][0]

        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        spam_score = spam_model.predict(padded)[0][0] 

        # Weigh the scores and return final prediction
        if phishing_score >= 0.5:
            return "Phishing"
        elif spam_score >= 0.5:
            return "Spam"
        else:
            return "Ham"
    else:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        spam_score = spam_model.predict(padded)[0][0]
        return "Spam" if spam_score >= 0.5 else "Ham"

def fine_tune_rnn(feedback_file='newspam.csv'):
    """
    Fine-tunes the RNN spam detection model using a feedback dataset.
    """
    try:
        # Load feedback dataset
        df_feedback = pd.read_csv(feedback_file, sep='\t', header=None, names=['label', 'message'])
        
        # Map labels for spam classification
        df_feedback['spam_label'] = df_feedback['label'].map({'spam': 1, 'ham': 0})
        df_feedback = df_feedback.dropna(subset=['spam_label', 'message'])

        if df_feedback.empty:
            raise ValueError("Feedback dataset is empty or contains invalid entries.")

        # Tokenize and pad sequences for spam
        sequences = tokenizer.texts_to_sequences(df_feedback['message'])
        padded_sequences = pad_sequences(sequences, maxlen=100)
        labels = df_feedback['spam_label'].values

        # Load spam detection model
        spam_model = load_model('spam_model.h5')

        # Fine-tune the spam model
        spam_model.fit(padded_sequences, labels, epochs=2, batch_size=16)
        
        # Save the updated spam model
        spam_model.save('spam_model.h5')
        print("Spam model retrained successfully.")

    except Exception as e:
        print(f"Error during spam model retraining: {str(e)}")


# Function to predict based on both spam and phishing models
def predict_final_output(test_text):
    urls = extract_url(test_text)
    
    # Case when there's no URL (just use the spam model)
    if len(urls) == 0:
        # Tokenize and pad the test text for the spam model
        test_text_tokenized = tokenizer.texts_to_sequences([test_text])
        test_text_padded = pad_sequences(test_text_tokenized, maxlen=100)  # Adjust maxlen as needed
        
        # Predict using the spam model
        spam_prediction = spam_model.predict(test_text_padded)[0]
        return spam_prediction
    
    # If there is a URL, use both models
    else:
        # Spam prediction on the entire text (including URL)
        test_text_tokenized = tokenizer.texts_to_sequences([test_text])
        test_text_padded = pad_sequences(test_text_tokenized, maxlen=100)
        spam_prediction = spam_model.predict(test_text_padded)[0]
        
        # Phishing prediction on the URL part only
        urls_tokenized = tokenizer.texts_to_sequences(urls)
        urls_padded = pad_sequences(urls_tokenized, maxlen=100)
        phishing_prediction = phishing_model.predict(urls_padded)[0]
        
        # Combine predictions with weights
        if spam_prediction == 1 and phishing_prediction == 1:
            final_prediction = 1 
        elif spam_prediction == 1:
            final_prediction = 0.4 
        elif phishing_prediction == 1:
            final_prediction = 0.6
        else:
            final_prediction = 0  # If both predict 0, it's benign/ham
        
        return final_prediction

def update_url_classifier(feedback_file='newphish.csv'):
    """
    Fine-tunes the phishing URL classifier using a feedback dataset.
    """
    try:
        # Load feedback dataset
        df_feedback = pd.read_csv(feedback_file, sep=',', header=0)  # Adjusted for CSV with header
        print(f"Dataset loaded. Shape: {df_feedback.shape}")
        print(df_feedback.head())

        # Remove rows with missing or invalid URLs
        df_feedback = df_feedback.dropna(subset=['url'])
        df_feedback['url'] = df_feedback['url'].astype(str).str.strip()
        print(f"After cleaning URLs. Shape: {df_feedback.shape}")

        # Map types for phishing classification
        label_mapping = {'phishing': 1, 'malware': 1, 'defacement': 1, 'benign': 0}
        df_feedback['phishing_label'] = df_feedback['type'].map(label_mapping)

        # Remove rows with invalid labels
        df_feedback = df_feedback.dropna(subset=['phishing_label'])
        print(f"After mapping labels. Shape: {df_feedback.shape}")
        print(df_feedback.head())

        if df_feedback.empty:
            raise ValueError("Feedback dataset is empty or contains invalid entries.")

        # Transform the URLs into hashed features
        hashed_feedback = hashing_vec.transform(df_feedback['url'])
        labels = df_feedback['phishing_label'].values

        # Load phishing URL classifier
        with open('phishing_model.pkl', 'rb') as f:
            phishing_classifier = pickle.load(f)

        # Update the phishing URL classifier
        phishing_classifier.partial_fit(hashed_feedback, labels, classes=[0, 1])
        
        # Save the updated phishing URL classifier
        with open('phishing_model.pkl', 'wb') as f:
            pickle.dump(phishing_classifier, f)
        print("Phishing URL classifier updated successfully.")

    except Exception as e:
        print(f"Error during phishing URL classifier update: {str(e)}")

if __name__ == "__main__":
    action = input("Enter action (predict/update): ").strip().lower()
    if action == "predict":
        test_text = input("Enter text: ").strip()
        result = predict_final_output(test_text)  # Function to handle both spam and phishing
        print(f"Result: {result}")
    elif action == "update":
        fine_tune_rnn(feedback_file='newspam.csv')
        update_url_classifier(feedback_file='newphish.csv')
        print("Models updated with feedback.")
