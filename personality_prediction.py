# personality_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_model():
    # Load data from CSV file
    data = pd.read_csv("train dataset.csv")

    # Split data into features (X) and target (y)
    X = data[['openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']]
    y = data['Personality (Class label)']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train a DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=0)  # You can adjust parameters as needed
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, "personality_model.pkl")

if __name__ == "__main__":
    train_model()
