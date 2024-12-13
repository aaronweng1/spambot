from flask import Flask, request, jsonify
import logging
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import HashingVectorizer
import re

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load the RNN spam model and tokenizer
spam_model = tf.keras.models.load_model('spam_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the URL classifier and HashingVectorizer
with open('phishing_model.pkl', 'rb') as f:
    url_classifier = pickle.load(f)
hashing_vec = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 4), n_features=5000)

# Function to detect URLs within text
def contains_url(text):
    return bool(re.search(r'http[s]?://|www\.|\.com|\.org|\.net|\.co|\.edu|\.gov|\.info|\.tv|\.me|\.wiki|\.in|\.eu|\.asia|\.xyz|\.ly|\.site', text))

@app.route('/classify', methods=['POST'])
def classify():
    global spam_model, url_classifier, tokenizer  # Ensure updates affect the loaded models

    # Reload the models dynamically
    spam_model = tf.keras.models.load_model('spam_model.h5')
    with open('phishing_model.pkl', 'rb') as f:
        url_classifier = pickle.load(f)
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    data = request.json
    text = data.get('text', '')
    url = data.get('url', '')

    response = {
        'combined_prediction': '',
        'spam_score': '',
        'phishing_score': ''
    }

    if text:
        if contains_url(text) or url:
            # URL classification (using the provided URL or extracting from text)
            url_to_classify = url if url else text
            url_features = hashing_vec.transform([url_to_classify])
            phishing_score = float(url_classifier.predict_proba(url_features)[:, 1][0])

            # Spam classification (using the full text)
            text_seq = tokenizer.texts_to_sequences([text])
            text_pad = pad_sequences(text_seq, maxlen=100)
            spam_score = float(spam_model.predict(text_pad)[0])

            # Weighted combined score
            url_weight = 0.8
            spam_weight = 0.2
            final_score = (spam_weight * spam_score) + (url_weight * phishing_score)
            final_prediction = 'Phishing' if final_score >= 0.5 else 'Ham'

            response['combined_prediction'] = final_prediction
            response['spam_score'] = spam_score
            response['phishing_score'] = phishing_score

        else:
            # If no URL, use only the spam model
            text_seq = tokenizer.texts_to_sequences([text])
            text_pad = pad_sequences(text_seq, maxlen=100)
            spam_score = float(spam_model.predict(text_pad)[0])
            spam_prediction = 'Spam' if spam_score >= 0.5 else 'Ham'

            response['combined_prediction'] = spam_prediction
            response['spam_score'] = spam_score

    elif url:
        # If only URL provided, classify with the URL model
        url_features = hashing_vec.transform([url])
        phishing_score = float(url_classifier.predict_proba(url_features)[:, 1][0])
        response['phishing_score'] = phishing_score
        response['combined_prediction'] = 'Phishing' if phishing_score >= 0.5 else 'Safe'

    if not response['combined_prediction']:
        return jsonify({'error': 'No prediction could be made'})

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
