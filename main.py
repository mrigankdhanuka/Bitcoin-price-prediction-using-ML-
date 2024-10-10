from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained LSTM model
model = load_model('bitcoin_lstm_model.h5')

@app.route('/')
def home():
    return "Welcome to the Bitcoin LSTM Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receive data in JSON format (input for prediction)
    sequence = np.array(data['sequence']).reshape(1, 60, 1)  # Reshape as per LSTM input
    prediction = model.predict(sequence)
    return jsonify({'predicted_price': prediction.tolist()})  # Return prediction

if __name__ == '__main__':
    app.run(debug=True, port=5000)
