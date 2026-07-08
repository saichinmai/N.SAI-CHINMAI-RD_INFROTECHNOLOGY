import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Get current folder path
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# CSV file path
csv_path = os.path.join(
    BASE_DIR,
    "sentiment_dataset.csv"
)

# Load dataset
data = pd.read_csv(
    csv_path
)

# Input and output
X = data["text"]
y = data["sentiment"]

# Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english"
)

X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()

model.fit(
    X_vectorized,
    y
)

# Save model
pickle.dump(
    model,
    open(
        "sentiment_model.pkl",
        "wb"
    )
)

pickle.dump(
    vectorizer,
    open(
        "vectorizer.pkl",
        "wb"
    )
)

print(
    "Model trained successfully!"
)