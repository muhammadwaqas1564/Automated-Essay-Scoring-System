# from flask import Flask,render_template,request,redirect
# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# app=Flask(__name__)
#
# @app.route("/",methods=["POST","GET"])
# def essay_prediction():
#     if request.method=="POST":
#         essay = request.form["essay"]
#         print(essay)
#         lemmatizer = WordNetLemmatizer()
#         stop_words = set(stopwords.words('english'))
#
#
#         text = essay.lower()
#         text = re.sub(r'[^\w\s]', '', text)
#         tokens = word_tokenize(text)
#         tokens = [word for word in tokens if word not in stop_words]
#         tokens = [lemmatizer.lemmatize(word) for word in tokens]
#         tokens= ' '.join(tokens)
#         print(tokens)
#
#
#     return render_template("input.html")
#
# if __name__=="__main__":
#     app.run(debug=


# from flask import Flask, render_template, request
# import spacy
# import re
# import joblib

# # Load spaCy's small English model
# nlp = spacy.load("en_core_web_sm")

# app = Flask(__name__)

# # Load the saved RandomForest model and TF-IDF Vectorizer
# rf_model = joblib.load('rf_model.joblib')
# tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# @app.route("/", methods=["POST", "GET"])
# def essay_prediction():
#     if request.method == "POST":
#         essay = request.form["essay"]
#         print(essay)

#         # Process the text using spaCy
#         doc = nlp(essay.lower())

#         # Remove stopwords, punctuation, and lemmatize the text
#         tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

#         # Join tokens with a single space
#         cleaned_text = ' '.join(tokens)

#         # Use regex to remove multiple spaces
#         cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

#         # Transform the cleaned text into a TF-IDF vector (do NOT fit the vectorizer again)
#         latest_input_tfidf = tfidf_vectorizer.transform([cleaned_text])

#         # Make a prediction using the loaded RandomForest model
#         prediction = rf_model.predict(latest_input_tfidf)
#         prediction[0]

#         # Print the prediction result
#         print("Prediction:", prediction[0])

#     return render_template("input.html",prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import re
import joblib
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")

# Ensure the model files are in the correct location
model_path = os.path.join(os.getcwd(), 'rf_model.joblib')
vectorizer_path = os.path.join(os.getcwd(), 'tfidf_vectorizer.joblib')

# Load the saved RandomForest model and TF-IDF Vectorizer
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    rf_model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
else:
    raise FileNotFoundError("Model or vectorizer file not found in the current directory.")

# Add a default route for GET requests
@app.route("/", methods=["GET"])
def home():
    return "Automated Essay Scoring API is running!"

@app.route("/predict", methods=["POST"])
def essay_prediction():
    if request.method == "POST":
        data = request.get_json()
        essay = data.get("essay", "")

        # Process the text using spaCy
        doc = nlp(essay.lower())

        # Remove stopwords, punctuation, and lemmatize the text
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

        # Join tokens with a single space
        cleaned_text = ' '.join(tokens)

        # Use regex to remove multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Transform the cleaned text into a TF-IDF vector
        latest_input_tfidf = tfidf_vectorizer.transform([cleaned_text])

        # Make a prediction using the loaded RandomForest model
        prediction = rf_model.predict(latest_input_tfidf)[0]

        # Return the prediction as JSON
        return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
