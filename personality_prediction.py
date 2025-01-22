from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_model():
    # Load training data
    train_data = pd.read_csv("train.csv")

    # Encode Gender if necessary
    train_data['Gender'] = train_data['Gender'].map({'Male': 0, 'Female': 1})

    # Split training data into features (X) and target (y)
    X_train = train_data[['Gender', 'Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']]
    y_train = train_data['Personality (class label)']

    # Train a DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=5, random_state=0)  # Adjust hyperparameters as needed
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, "personality_model.pkl")

    print("Model trained and saved as personality_model.pkl.")

def evaluate_model():
    # Load the trained model
    model = joblib.load("personality_model.pkl")

    # Load test data
    test_data = pd.read_csv("test.csv")

    # Encode Gender if necessary
    test_data['Gender'] = test_data['Gender'].map({'Male': 0, 'Female': 1})

    # Split test data into features (X) and target (y)
    X_test = test_data[['Gender', 'Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']]
    y_test = test_data['Personality (class label)']

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_model()
    evaluate_model()
