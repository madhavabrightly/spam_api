from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('spam_detector_model1.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    result = model.predict([text])[0]
    return jsonify({'spam': bool(result)})

@app.route('/')
def home():
    return "Spam Detector API Running"

if __name__ == '__main__':
    app.run(debug=True)
