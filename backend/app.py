# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load the trained model and the class names
model = joblib.load('iris_model.pkl')
iris_target_names = ['setosa', 'versicolor', 'virginica']

app = Flask(__name__)
# Enable CORS to allow requests from the front end
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features from the JSON data
        sepal_length = data['sepal_length']
        sepal_width = data['sepal_width']
        petal_length = data['petal_length']
        petal_width = data['petal_width']

        # Create a NumPy array for prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make the prediction
        prediction_index = model.predict(features)[0]
        prediction_name = iris_target_names[prediction_index]

        return jsonify({'prediction': prediction_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)