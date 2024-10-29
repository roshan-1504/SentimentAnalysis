import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle


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

# 4. Tokenizing and padding sequences for CNN input
vocab_size = 10000  # Max vocabulary size
max_length = 100    # Max sequence length

# Tokenizer for converting text to integer sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Convert the text data to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding the sequences to ensure they have the same length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# One-hot encode the target labels
y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

# 5. Building the CNN model
embedding_dim = 100  # Dimension of the embedding layer

cnn_model = Sequential()

# Embedding layer (removed input_length parameter)
cnn_model.add(Embedding(vocab_size, embedding_dim))

# Convolutional layer
cnn_model.add(Conv1D(128, 5, activation='relu'))

# Pooling layer
cnn_model.add(GlobalMaxPooling1D())

# Dense layer
cnn_model.add(Dense(128, activation='relu'))

# Output layer
cnn_model.add(Dense(2, activation='softmax'))

# Compile the model
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Training the CNN model
cnn_model.fit(X_train_pad, y_train_cat, epochs=5, batch_size=64, validation_split=0.1)

# 7. Making predictions on the test data
y_pred_cnn = cnn_model.predict(X_test_pad)

# Convert predictions to binary labels
y_pred_cnn_labels = np.argmax(y_pred_cnn, axis=1)

# 8. Evaluating the CNN model
accuracy_cnn = accuracy_score(y_test, y_pred_cnn_labels)
print(f"CNN Accuracy: {accuracy_cnn * 100:.2f}%")

# Classification report for detailed metrics
print(classification_report(y_test, y_pred_cnn_labels, target_names=['Negative', 'Positive']))

# 9. Predict sentiment for a new text
def predict_sentiment_cnn(input_text, model, tokenizer, max_len):
    # Clean the input text
    clean_input_text = clean_text(input_text)
    
    # Convert to sequences and pad
    input_seq = tokenizer.texts_to_sequences([clean_input_text])
    input_pad = pad_sequences(input_seq, maxlen=max_len, padding='post', truncating='post')
    
    # Predict sentiment using the CNN model
    prediction = model.predict(input_pad)
    
    # Return sentiment result
    return 'Positive' if np.argmax(prediction) == 1 else 'Negative'

# Example of predicting sentiment for a new text
input_text = "I really love this series, it's fantastic!"
result = predict_sentiment_cnn(input_text, cnn_model, tokenizer, max_length)
print(f"The sentiment of the given text is: {result}")

# Save the model in the new Keras format
cnn_model.save('model_CNN.keras')

# After fitting the tokenizer
with open('tokenizer_CNN.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)