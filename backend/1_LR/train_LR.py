import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import string
import joblib

# Load dataset (you may need to adjust the file path)
df = pd.read_csv('../Sentiment140.csv', encoding='ISO-8859-1', header=None)

# Assigning column names
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Check for null values
df.isnull().sum()

# Remove unwanted columns
df = df[['target', 'text']]

# Mapping sentiment 0 to negative and 4 to positive
df['target'] = df['target'].map({0: 0, 4: 1})

# Function to clean the text data
def clean_text(text):
    # Removing URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    
    return text

# Apply text cleaning
df['text'] = df['text'].apply(clean_text)

# Preview cleaned data
print(df.head())

# Split dataset into features (X) and target (y)
X = df['text']
y = df['target']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf.transform(X_test)

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train_tfidf, y_train)

# Predict the test data
y_pred = model.predict(X_test_tfidf)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

def predict_sentiment(input_text, model, vectorizer):
    # Clean the input text
    clean_input_text = clean_text(input_text)
    
    # Transform the text using the same TF-IDF vectorizer used during training
    input_tfidf = vectorizer.transform([clean_input_text])
    
    # Predict the sentiment using the trained logistic regression model
    prediction = model.predict(input_tfidf)
    
    # Return the result
    return 'Positive' if prediction[0] == 1 else 'Negative'


# Example text for sentiment prediction
# input_text = "I'm super bad"

# Predict sentiment for the example text
# result = predict_sentiment(input_text, model, tfidf)
# print(f"The sentiment of the given text is: {result}")


# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Define your model here
model.fit(X_train_tfidf, y_train)

# Save the model
# joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(model, 'model_LR.pkl')

# Save the vectorizer
# joblib.dump(tfidf, 'lrm_tfidf_vectorizer.pkl')
joblib.dump(tfidf, 'vectorizer_LR.pkl')