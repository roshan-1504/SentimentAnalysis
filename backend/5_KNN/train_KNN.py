# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load the dataset
# Assuming the file is named 'Sentiment140.csv'
df = pd.read_csv('../Sentiment140.csv', encoding='ISO-8859-1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Step 2: Preprocess the data
# For simplicity, we will only use the 'text' and 'target' columns
X = df['text']
y = df['target']

# Convert the target values from [0, 4] to [0, 1] (0 = Negative, 1 = Positive)
y = y.apply(lambda x: 1 if x == 4 else 0)

# Step 3: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)  # Limit the number of features for faster training
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 5: Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train_tfidf, y_train)

# Step 6: Make predictions
y_pred_knn = knn_model.predict(X_test_tfidf)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy * 100:.2f}%")
# Classification report to see precision, recall, f1-score
print("Classification Report:")
print(classification_report(y_test, y_pred_knn))

# Step 8: Save the trained KNN model and TF-IDF vectorizer
with open('model_KNN.pkl', 'wb') as model_file:
    pickle.dump(knn_model, model_file)
with open('vectorizer_KNN.pkl', 'wb') as vec_file:
    pickle.dump(tfidf, vec_file)

# Load the saved model and vectorizer
with open('model_KNN.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

with open('vectorizer_KNN.pkl', 'rb') as vec_file:
    tfidf_vectorizer = pickle.load(vec_file)

# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess the input text using the saved TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([text])
    
    # Predict using the saved KNN model
    prediction = knn_model.predict(text_vectorized)
    
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Example prediction
test_text = "I don't know!"
print(f"Predicted sentiment: {predict_sentiment(test_text)}")