import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import string
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the Sentiment140 dataset
data = pd.read_csv('../Sentiment140.csv', encoding='latin-1', usecols=[0, 5], names=['target', 'text'])
data['target'] = data['target'].map({0: 0, 4: 1})  # Convert to binary (0: Negative, 1: Positive)

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.lower()  # Convert to lowercase

# Clean the text data
data['text'] = data['text'].apply(clean_text)

# Split data into features and target
X = data['text']
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=10000)  # You can adjust max_features as needed

# Fit and transform the training data, and transform the testing data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create and train the LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train_tfidf, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'model_LGBM.pkl')
joblib.dump(vectorizer, 'vectorizer_LGBM.pkl')

import joblib
import re
import string

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

# Load the saved model and vectorizer
model = joblib.load('model_LGBM.pkl')
vectorizer = joblib.load('vectorizer_LGBM.pkl')

# Function to predict sentiment for new statements
def predict_sentiment(text):
    # Clean and transform the input text using the loaded TF-IDF vectorizer
    clean_input_text = clean_text(text)
    text_vectorized = vectorizer.transform([clean_input_text])
    
    # Predict using the saved LightGBM model
    prediction = model.predict(text_vectorized)
    
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Example usage
if __name__ == '__main__':
    test_text = "I love this product"
    predicted_sentiment = predict_sentiment(test_text)
    print(predicted_sentiment)