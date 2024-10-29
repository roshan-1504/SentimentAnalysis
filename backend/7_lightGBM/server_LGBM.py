from flask import Flask, request, jsonify
import joblib
import re
import string

# Initialize the Flask application
app = Flask(__name__)

# Load the saved LightGBM model and TF-IDF vectorizer
model = joblib.load('model_LGBM.pkl')
vectorizer = joblib.load('vectorizer_LGBM.pkl')

# Function to clean the input text
def clean_text(text):
    # Removing URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Removing mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

# Function to predict sentiment using LightGBM model
def predict_sentiment(text):
    # Clean and transform the input text using the loaded TF-IDF vectorizer
    clean_input_text = clean_text(text)
    text_vectorized = vectorizer.transform([clean_input_text])
    
    # Predict using the saved LightGBM model
    prediction = model.predict(text_vectorized)
    
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Define the predict endpoint for the sentiment analysis
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
    result = predict_sentiment(input_text)
    
    # Log the prediction result
    print("Prediction result:", result)
    
    # Return the result as JSON
    return jsonify({'prediction': result})

# Run the Flask app on port 5007
if __name__ == '__main__':
    app.run(port=5007)
