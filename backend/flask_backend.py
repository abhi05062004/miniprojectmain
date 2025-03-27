# from flask import Flask, request, jsonify
# import pickle
# from ingredient_substitution_model import EnhancedIngredientSubstitution  # Import the class
# from collections import defaultdict
# import os

# app = Flask(__name__)

# # Define the defaultdict_int function
# def defaultdict_int():
#     return defaultdict(int)

# # Load the trained model
# model_path = os.path.join(os.path.dirname(__file__), 'enhanced_ingredient_substitution.pkl')
# with open(model_path, 'rb') as file:
#     system = pickle.load(file)

# @app.route('/substitutes', methods=['POST'])
# def get_substitutes():
#     """API endpoint to get ingredient substitutes."""
#     data = request.json
#     ingredient = data.get('ingredient')
#     cuisine = data.get('cuisine', None)
#     if not ingredient:
#         return jsonify({'error': 'Ingredient is required'}), 400
#     substitutes = system.find_substitutes(ingredient, cuisine)
#     return jsonify({'ingredient': ingredient, 'substitutes': substitutes})

# if __name__ == '__main__':
#     app.run(debug=True)


import numpy as np
import tensorflow as tf
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Load the updated dataset
file_path = "backend/updated_food1.csv"
data = pd.read_csv(file_path)

# Remove spaces from column names (if needed)
data.columns = data.columns.str.strip()

# Load the saved LabelEncoder classes
encoder_classes = np.load("backend/label_encoder_classes.npy", allow_pickle=True)
encoder = LabelEncoder()
encoder.classes_ = encoder_classes

# Load trained model
model = tf.keras.models.load_model("backend/food_substitution_model.h5")

# Function to get food alternative
def get_alternative(food_name):
    if food_name not in encoder.classes_:
        return {'error': f"'{food_name}' is not in the dataset! No substitute found."}
    
    # Convert food name to numerical ID
    food_id = encoder.transform([food_name])[0]
    
    # Predict alternative food
    predictions = model.predict(np.array([food_id]).reshape(1, 1))
    alternative_id = np.argmax(predictions)
    
    # Get alternative food name
    alternative_food = encoder.inverse_transform([alternative_id])[0]
    
    return {'ingredient': food_name, 'substitute': alternative_food}

@app.route('/substitute', methods=['POST'])
def get_substitutes():
    data = request.json
    ingredient = data.get('ingredient')
    
    if not ingredient:
        return jsonify({'error': 'Ingredient is required'}), 400
    
    result = get_alternative(ingredient.lower())
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
