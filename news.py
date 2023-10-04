#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib  # Import joblib to load the model

from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("fakeNewsModel.pkl")  # Replace with the actual path to your trained model file

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Replace with the actual path to your TF-IDF vectorizer file

# Function to perform stemming
ps = PorterStemmer()
def stemming(title):
    stemmed_title = re.sub('[^a-zA-Z]', ' ', title)
    stemmed_title = stemmed_title.lower()
    stemmed_title = stemmed_title.split()
    stemmed_title = [ps.stem(word) for word in stemmed_title if not word in stopwords.words('english')]
    stemmed_title = ' '.join(stemmed_title)
    return stemmed_title

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']  # Get the input from the form

    # Perform stemming and vectorization
    stemmed_input = stemming(news_text)
    input_vector = vectorizer.transform([stemmed_input])

    # Make a prediction using the model
    prediction = model.predict(input_vector)

    result = "Fake News" if prediction[0] == 1 else "Real News"
    return render_template('result.html', result=result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False)
