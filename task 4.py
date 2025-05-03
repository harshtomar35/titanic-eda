# Importing necessary libraries
# Importing necessary libraries
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK stopwords (if needed)
nltk.download('stopwords')

# Load the dataset (replace 'sentiment_data.csv' with the actual file path)
# For example: df = pd.read_csv('path_to_your_dataset.csv')
df = pd.read_csv('twitter_training.csv')

# Check the first few rows of the dataset to understand the structure
print("Dataset Columns:", df.columns)
print(df.head())

# Checking the column names and ensuring they exist in the dataset
# This will help to identify the correct column names
print("Columns available in dataset:", df.columns)

# Preprocessing: Remove null values, if any
df.dropna(subset=['text', 'sentiment'], inplace=True)

# Encode sentiment labels (assuming 'positive', 'negative', 'neutral' labels in the 'sentiment' column)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})

# Verify that the sentiment column is now correctly encoded
print("Sentiment column unique values:", df['sentiment'].unique())

# Split the data into features (X) and target (y)
X = df['text']  # Features: Text data (replace with the correct column name if necessary)
y = df['sentiment']  # Target: Sentiment labels

# Split the dataset into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical data using TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train_tfidf, y_train)

# Predict on the validation set
y_pred = model.predict(X_val_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display Classification Report
print('Classification Report:')
print(classification_report(y_val, y_pred))

# Confusion Matrix visualization
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the model (Optional)
import joblib
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved!")
