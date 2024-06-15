from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("personality_model.pkl")


# Define the route for the home page with the form
@app.route("/", methods=["GET"])
def home():
    return render_template("index1.html")


# Define the route for receiving input and making predictions
@app.route("/predict", methods=["POST"])
def predict_personality():
    try:
        # Get the OCEAN traits from the form submission
        traits = [float(request.form["openness"]),
                  float(request.form["conscientiousness"]),
                  float(request.form["extraversion"]),
                  float(request.form["agreeableness"]),
                  float(request.form["neuroticism"])]

        # Make predictions using the loaded model
        prediction = model.predict([traits])[0]

        # Render the result template with the predicted personality
        return render_template("result.html", personality=prediction)
    except Exception as e:
        return render_template("error.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
