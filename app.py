from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_FILE = "personality_model.pkl"

# Route: Home (index.html)
@app.route("/")
def index():
    return render_template("index.html")

# Route: Predict Personality
@app.route("/predict", methods=["POST"])
def predict_personality():
    try:
        # Extract form data
        gender = request.form["gender"]
        age = int(request.form["age"])
        openness = int(request.form["openness"])
        conscientiousness = int(request.form["conscientiousness"])
        extraversion = int(request.form["extraversion"])
        agreeableness = int(request.form["agreeableness"])
        neuroticism = int(request.form["neuroticism"])

        # Encode Gender
        gender_map = {"male": 0, "female": 1, "other": 2}
        if gender not in gender_map:
            return render_template("error.html", message="Invalid gender selection.")
        gender_encoded = gender_map[gender]

        # Load the trained model
        if not os.path.exists(MODEL_FILE):
            return render_template("error.html", message="Model file not found. Train the model first.")
        model = joblib.load(MODEL_FILE)

        # Prepare data for prediction
        input_features = [[gender_encoded, age, openness, neuroticism, conscientiousness, agreeableness, extraversion]]
        prediction = model.predict(input_features)[0]

        # Render the result page
        return render_template("result.html", personality=prediction)

    except Exception as e:
        return render_template("error.html", message=f"An error occurred: {str(e)}")

# Route: Train Model
@app.route("/train", methods=["GET", "POST"])
def train_model():
    if request.method == "POST":
        try:
            # Load training data
            train_data = pd.read_csv("train.csv")
            train_data["Gender"] = train_data["Gender"].map({"Male": 0, "Female": 1, "Other": 2})
            if train_data["Gender"].isnull().any():
                return render_template("error.html", message="Invalid Gender values in train.csv.")

            # Split data into features and labels
            X_train = train_data[["Gender", "Age", "openness", "neuroticism", "conscientiousness", "agreeableness", "extraversion"]]
            y_train = train_data["Personality (class label)"]

            # Train the model
            model = DecisionTreeClassifier(max_depth=5, random_state=0)
            model.fit(X_train, y_train)

            # Save the model
            joblib.dump(model, MODEL_FILE)
            return render_template("result.html", personality="Model trained successfully.")

        except FileNotFoundError:
            return render_template("error.html", message="train.csv file not found.")
        except Exception as e:
            return render_template("error.html", message=f"An error occurred during training: {str(e)}")
    return render_template("train.html")

if __name__ == "__main__":
    app.run(debug=True)
