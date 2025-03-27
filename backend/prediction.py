import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from ultralytics import YOLO
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'backend/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load dataset
file_path = "backend/updated_food1.csv"
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()  # Remove spaces from column names

# Load LabelEncoder classes
encoder_classes = np.load("backend/label_encoder_classes.npy", allow_pickle=True)
encoder = LabelEncoder()
encoder.classes_ = encoder_classes

# Load trained models
substitution_model = tf.keras.models.load_model("backend/food_substitution_model.h5")
yolo_model = YOLO("backend/vegetable.pt")  # Load YOLO model

# Function to get food alternative
def get_alternative(food_name):
    if food_name not in encoder.classes_:
        return {'error': f"'{food_name}' is not in the dataset! No substitute found."}

    # Convert food name to numerical ID
    food_id = encoder.transform([food_name])[0]

    # Predict alternative food
    predictions = substitution_model.predict(np.array([food_id]).reshape(1, 1))
    alternative_id = np.argmax(predictions)

    # Get alternative food name
    alternative_food = encoder.inverse_transform([alternative_id])[0]

    return {'ingredient': food_name, 'substitute': alternative_food}

# Endpoint to get substitutes from text input
@app.route('/substitute', methods=['POST'])
def get_substitutes():
    data = request.json
    ingredient = data.get('ingredient')

    if not ingredient:
        return jsonify({'error': 'Ingredient is required'}), 400

    result = get_alternative(ingredient.lower())
    return jsonify(result)

# Endpoint to predict vegetable from image and find substitutes
@app.route('/predict', methods=['POST'])
def predict_vegetable():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Save the uploaded image
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read and process image
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for YOLO

    # Run YOLO model for vegetable detection
    results = yolo_model(img)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return jsonify({'error': 'No vegetable detected'}), 404

    # Get detected vegetable name
    detected_vegetable = results[0].names[int(results[0].boxes.cls[0])]

    # Find a substitute for detected vegetable
    substitute_response = get_alternative(detected_vegetable.lower())

    return jsonify({'detected_vegetable': detected_vegetable, **substitute_response})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
