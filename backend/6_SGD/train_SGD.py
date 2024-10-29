# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset
data = pd.read_csv('../Sentiment140.csv', encoding='latin-1', header=None)

# Step 2: Rename the columns (as per the structure of the Sentiment140 dataset)
data.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Step 3: Prepare the data (only keeping the necessary columns: 'text' and 'target')
data = data[['text', 'target']]

# Step 4: Convert target to binary (positive = 1, negative = 0)
data['target'] = data['target'].apply(lambda x: 1 if x == 4 else 0)

# Step 5: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

# Step 6: Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 7: Initialize the SGDClassifier (Stochastic Gradient Descent)
sgd_clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42, n_jobs=-1)

# Step 8: Train the model
sgd_clf.fit(X_train_tfidf, y_train)


# Step 9: Make predictions on the test set
y_pred = sgd_clf.predict(X_test_tfidf)

# Step 10: Calculate accuracy on the test set
y_pred = sgd_clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%") 

# Step 11: Evaluate the model
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

import pickle
with open('model_SGD.pkl', 'wb') as model_file:
    pickle.dump(sgd_clf, model_file)
with open('vectorizer_SGD.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

import pickle

# Load the trained SGD model and TF-IDF vectorizer
with open('model_SGD.pkl', 'rb') as model_file:
    sgd_model = pickle.load(model_file)

with open('vectorizer_SGD.pkl', 'rb') as vec_file:
    tfidf_vectorizer = pickle.load(vec_file)

# Function to predict sentiment of a new statement
def predict_sentiment(text):
    # Vectorize the input text using the loaded TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([text])
    
    # Predict the sentiment using the loaded SGD model
    prediction = sgd_model.predict(text_vectorized)
    
    # Return the sentiment result
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Example: Predict sentiment for a new statement
new_statement = "I love the way this app works, absolutely amazing!"
predicted_sentiment = predict_sentiment(new_statement)

print(predicted_sentiment)