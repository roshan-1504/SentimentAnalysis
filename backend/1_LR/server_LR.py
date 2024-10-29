from flask import Flask, request, jsonify
import joblib
import re
import string

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('model_LR.pkl')
tfidf = joblib.load('vectorizer_LR.pkl')

def predict_sentiment(input_text, model, vectorizer):
    # Clean the input text
    clean_input_text = clean_text(input_text)
    
    # Transform the text using the same TF-IDF vectorizer used during training
    input_tfidf = vectorizer.transform([clean_input_text])
    
    # Predict the sentiment using the trained logistic regression model
    prediction = model.predict(input_tfidf)
    
    # Return the result
    return 'Positive' if prediction[0] == 1 else 'Negative'

def clean_text(text):
    # Removing URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    
    return text

# Define the predict endpoint
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
    result = predict_sentiment(input_text, model, tfidf)
    
    # Log the prediction result
    print("Prediction result:", result)
    
    # Return the result as JSON
    return jsonify({'prediction': result})

# Run the Flask app on port 5001
if __name__ == '__main__':
    app.run(port=5001)
