from flask import Flask, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import string

# Initialize the Flask application
app = Flask(__name__)

# Load the saved CNN model and tokenizer
cnn_model = load_model('model_CNN.keras')
with open('tokenizer_CNN.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define max_length to match what you used during training
max_length = 100  # This should match what you used during training

# Function to clean the input text
def clean_text(text):
    # Removing URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    
    return text

# Function to predict sentiment using CNN model
def predict_sentiment_cnn(input_text, model, tokenizer, max_len):
    # Clean the input text
    clean_input_text = clean_text(input_text)
    
    # Convert the text to sequences and pad it
    input_seq = tokenizer.texts_to_sequences([clean_input_text])
    
    # Handle empty sequences
    if len(input_seq[0]) == 0:
        return "Unable to tokenize input text."
    
    input_pad = pad_sequences(input_seq, maxlen=max_len, padding='post', truncating='post')
    
    # Make the prediction using the CNN model
    prediction = model.predict(input_pad)
    
    # Return the result
    return 'Positive' if np.argmax(prediction) == 1 else 'Negative'

# Define the predict endpoint for the CNN model
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
    result = predict_sentiment_cnn(input_text, cnn_model, tokenizer, max_length)
    
    # Log the prediction result
    print("Prediction result:", result)
    
    # Return the result as JSON
    return jsonify({'prediction': result})

# Run the Flask app on port 5002
if __name__ == '__main__':
    app.run(port=5002)
