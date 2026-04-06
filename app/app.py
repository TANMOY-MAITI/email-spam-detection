from flask import Flask, request, jsonify, render_template
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('../model/spam_model.pkl')
tfidf = joblib.load('../model/tfidf_vectorizer.pkl')

# Preprocessing — same function as notebook
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')

    if not message.strip():
        return jsonify({'error': 'No message provided'}), 400

    cleaned = preprocess(message)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    spam_prob = round(float(probability[1]) * 100, 2)
    ham_prob = round(float(probability[0]) * 100, 2)

    return jsonify({
        'prediction': prediction,
        'spam_probability': spam_prob,
        'ham_probability': ham_prob,
        'message': message
    })

if __name__ == '__main__':
    app.run(debug=True)