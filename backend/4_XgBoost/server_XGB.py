from flask import Flask, request, jsonify
import joblib
import re
import string

# Initialize the Flask application
app = Flask(__name__)

# Load the saved XGBoost model and TF-IDF vectorizer
model = joblib.load('model_XGB.pkl')
vectorizer = joblib.load('vectorizer_XGB.pkl')

# Function to clean the input text
def clean_text(text):
    # Removing URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    
    return text

# Function to predict sentiment using XGBoost model
def predict_sentiment_xgb(input_text, model, vectorizer):
    # Clean the input text
    clean_input_text = clean_text(input_text)
    
    # Transform the text using the TF-IDF vectorizer
    input_tfidf = vectorizer.transform([clean_input_text])
    
    # Predict sentiment using the XGBoost model
    prediction = model.predict(input_tfidf)
    
    # Return sentiment result
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Define the predict endpoint for the XGBoost model
@app.route('/predict', methods=['POST'])
def predict():
    # Log the request data
    print("Received data:", request.get_json())
    
    # Get the input text from the request
    data = request.get_json()
    input_text = data.get('statement', '')
    
    # If no text is provided, return an error
    if not input_text:
        print("Error: No text provided.")
        return jsonify({'error': 'No text provided'}), 400

    # Predict the sentiment
    result = predict_sentiment_xgb(input_text, model, vectorizer)
    
    # Log the prediction result
    print("Prediction result:", result)
    
    # Return the result as JSON
    return jsonify({'prediction': result})

# Run the Flask app on port 5003
if __name__ == '__main__':
    app.run(port=5004)
