import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Expected feature names order from standard Kaggle Heart Disease dataset
FEATURE_NAMES = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal',
]


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'API is running',
        'message': (
            'Send a POST request to /predict with patient clinical parameters.'
        ),
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract features in the correct order
        features = [data[feature] for feature in FEATURE_NAMES]
        final_features = np.array([features])

        # Make prediction
        prediction = model.predict(final_features)[0]

        # Map binary outcome to descriptive response
        result = (
            "Heart Disease Detected"
            if prediction == 1
            else "No Heart Disease Detected"
        )

        return jsonify({'prediction': result})

    except KeyError as e:
        return jsonify({'error': f'Missing required feature: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)