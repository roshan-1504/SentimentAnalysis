from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
import string
from langdetect import detect

# Load the trained model and vectorizer
model_CNB = joblib.load('model_CNB.pkl')
vectorizer_CNB = joblib.load('vectorizer_CNB.pkl')


# Initialize Flask app
app = Flask(__name__)

# Preprocessing functions (as provided in the original code)
def remove_numbers(text):
    return ''.join(char for char in text if not char.isdigit()) if isinstance(text, str) else text

def remove_character(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
        text = re.sub(r"\b\d+\b", "", text)  # remove numbers
        text = re.sub(r'<.*?>+', '', text)  # remove HTML tags
        text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)  # remove punctuation
        text = re.sub(r'\n', '', text)  # remove newlines
        text = re.sub(r'[’“”…]', '', text)  # remove special characters
        return text
    else:
        return text

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # geometric shapes extended
                               u"\U0001F800-\U0001F8FF"  # supplemental arrows
                               u"\U0001F900-\U0001F9FF"  # supplemental symbols
                               u"\U00002702-\U000027B0"  # dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_short_form(text):
    if isinstance(text, str):
        text = re.sub(r"isn't", 'is not', text)
        text = re.sub(r"he's", 'he is', text)
        text = re.sub(r"wasn't", 'was not', text)
        text = re.sub(r"there's", 'there is', text)
        text = re.sub(r"couldn't", 'could not', text)
        text = re.sub(r"won't", 'will not', text)
        text = re.sub(r"they're", 'they are', text)
        text = re.sub(r"she's", 'she is', text)
        text = re.sub(r"wouldn't", 'would not', text)
        text = re.sub(r"haven't", 'have not', text)
        text = re.sub(r"that's", 'that is', text)
        text = re.sub(r"you've", 'you have', text)
        text = re.sub(r"what's", 'what is', text)
        text = re.sub(r"weren't", 'were not', text)
        text = re.sub(r"we're", 'we are', text)
        text = re.sub(r"hasn't", 'has not', text)
        text = re.sub(r"shouldn't", 'should not', text)
        text = re.sub(r"let's", 'let us', text)
        text = re.sub(r"i'm", 'i am', text)
        text = re.sub(r"it's", 'it is', text)
        text = re.sub(r"don't", 'do not', text)
        text = re.sub(r"i’d", 'i did', text)
        return text
    else:
        return text

def remove_multiple_space(text):
    return re.sub(r'\s+', ' ', text.strip()) if isinstance(text, str) else text

# Predict sentiment function
def predict_sentiment(input_text):
    # Preprocess the input_text
    input_text_cleaned = remove_numbers(input_text)
    input_text_cleaned = remove_character(input_text_cleaned)
    input_text_cleaned = remove_emoji(input_text_cleaned)
    input_text_cleaned = remove_short_form(input_text_cleaned)
    input_text_cleaned = remove_multiple_space(input_text_cleaned)

    # Convert input text into numerical format using CountVectorizer
    input_text_vectorized =vectorizer_CNB.transform([input_text_cleaned])

    # Predict sentiment (0 = Negative, 1 = Positive)
    prediction = model_CNB.predict(input_text_vectorized)

    # Return the prediction result
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Define a route for prediction
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
    result = predict_sentiment(input_text)
    
    # Log the prediction result
    print("Prediction result:", result)
    
    # Return the result as JSON
    return jsonify({'prediction': result})

# Run the server
if __name__ == '__main__':
    app.run(port=5003)
