import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load data
data = pd.read_csv("medical_data.csv")

# Features and target
X = data.drop("Disease", axis=1)
y = data["Disease"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("disease_model.pkl", "wb"))

print("Model trained successfully!")