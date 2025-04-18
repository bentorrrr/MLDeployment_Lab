from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
import os

if(os.path.exists("./model.pkl")):
    with open("./model.pkl", "rb") as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError("The file './model.pkl' was not found. Please ensure the model file is in the correct directory.")
  

if(os.path.exists("./housingmodel.pkl")):
    with open("./housingmodel.pkl", "rb") as f:
        housingModel = pickle.load(f)
else:
    raise FileNotFoundError("The file './housingmodel.pkl' was not found. Please ensure the model file is in the correct directory.")

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    if "features" not in data:
      return jsonify({"error": "Missing 'features' key"}), 400

    features = data["features"]
    
    if not isinstance(features, list):
        return jsonify({"error": "'features' must be a list"}), 400
      
    for item in features:
        if not (isinstance(item, list) and len(item) == 4 and all(isinstance(i, (int, float)) for i in item)):
            return jsonify({"error": "Each feature must be a list of 4 numbers"}), 400
    
    input_features = np.array(data["features"])
    predictions = model.predict(input_features)
    confidences = model.predict_proba(input_features).max(axis=1).tolist()
    
    # You can return an array of predictions & confidences
    return jsonify({
        "predictions": predictions.tolist(),
        "confidences": confidences
    })
    
EXPECTED_FEATURES = 13    
@app.route("/predicthousing", methods=["POST"])
def predict_housing():
    data = request.get_json()

    if "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400

    features = data["features"]

    if not isinstance(features, list):
        return jsonify({"error": "'features' must be a list"}), 400

    for i, item in enumerate(features):
        if not (isinstance(item, list) and len(item) == EXPECTED_FEATURES and all(isinstance(i, (int, float)) for i in item)):
            return jsonify({
                "error": f"Each feature must be a list of {EXPECTED_FEATURES} numbers. Error at index {i}"
            }), 400

    input_features = np.array(features)
    predictions = housingModel.predict(input_features)

    return jsonify({
        "predictions": predictions.tolist()
    })


@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9000) #check your port number ( if it is in use, change the port number)
