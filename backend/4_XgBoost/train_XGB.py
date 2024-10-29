# Importing necessary libraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# 1. Loading the Sentiment140 dataset
df = pd.read_csv('../Sentiment140.csv', encoding='ISO-8859-1', header=None)

# Assigning column names to the dataset
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Remove unnecessary columns
df = df[['target', 'text']]

# Map the target values (0 = negative, 4 = positive)
df['target'] = df['target'].map({0: 0, 4: 1})

# 2. Preprocessing the text data
def clean_text(text):
    # Removing URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    
    return text

# Apply the text cleaning function to the dataset
df['text'] = df['text'].apply(clean_text)

# 3. Splitting the dataset into training and testing sets
X = df['text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Vectorizing the text data using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf.transform(X_test)

# 5. Training the XGBoost model
xgb_model = xgb.XGBClassifier(eval_metric='logloss')  # Remove 'use_label_encoder'

# Fit the model on the training data
xgb_model.fit(X_train_tfidf, y_train)

# 6. Making predictions on the test data
y_pred_xgb = xgb_model.predict(X_test_tfidf)

# 7. Evaluating the XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb * 100:.2f}%")

# Classification report for detailed metrics
print(classification_report(y_test, y_pred_xgb, target_names=['Negative', 'Positive']))

# 8. Predict sentiment for a new text
def predict_sentiment_xgb(input_text, model, vectorizer):
    # Clean the input text
    clean_input_text = clean_text(input_text)
    
    # Transform the text using the TF-IDF vectorizer
    input_tfidf = vectorizer.transform([clean_input_text])
    
    # Predict sentiment using the XGBoost model
    prediction = model.predict(input_tfidf)
    
    # Return sentiment result
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Example of predicting sentiment for a new text
input_text = "I really hate this movie, it's beautiful!"
result = predict_sentiment_xgb(input_text, xgb_model, tfidf)
print(f"The sentiment of the given text is: {result}")

import joblib
joblib.dump(xgb_model, 'xgboost_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')