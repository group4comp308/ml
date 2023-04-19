from flask import Flask, jsonify, request
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model and columns
model = joblib.load("heart_disease_model.pkl")
columns = joblib.load("heart_disease_columns.pkl")

# Define the prediction function
def predict_heart_disease(data):
    try:
        # Create a numpy array from the input data
        input_data = np.array(data)

        # Reshape the input data to have a shape of (1, n)
        input_data = input_data.reshape(1, -1)

        # Make the prediction and extract the predicted classes
        prediction = model.predict(input_data)
        predicted_classes = [int(p[0] > 0.5) for p in prediction]
        print(predicted_classes)

        # Return the predicted classes as a dictionary
        return {"prediction": predicted_classes}
    except Exception as e:
        app.logger.error(str(e), exc_info=True)
        return {"error": "An error occurred during prediction."}

# Define the route for the prediction API
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    data = request.json["data"]
    
    # Make the prediction
    prediction = predict_heart_disease(data)
    
    # Return the prediction as JSON
    return jsonify(prediction)

# Define a home page
@app.route("/")
def home():
    return "<h1>Heart Disease Prediction API</h1>"

if __name__ == "__main__":
    app.run()
