import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")

# Split features & target (replace 'target' with your real column name)
X = df.drop("Category", axis=1)
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Save trained model
with open("Liver2.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as Liver2.pkl")

import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("Liver2.pkl", "rb"))

# Define feature names in the same order as training
feature_names = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 
                 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT', 'Sex_m']

# Test sample
X_test_sample = pd.DataFrame([[36,46.0,39.3,67.1,161.9,13.0,9.24,4.81,65.3,60.0,73.9,0.0]],
                             columns=feature_names)

# Predict
print(model.predict(X_test_sample))
