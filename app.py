import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import LabelEncoder
import os

# ----------------------------------------
# Flask App Configurations
# ----------------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')

# ----------------------------------------
# Load Trained Model and Tokenizer
# ----------------------------------------
MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.json"
LABEL_CLASSES_PATH = "label_classes.npy"
MAX_SEQUENCE_LENGTH = 50

# Load model
MODEL = load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# ----------------------------------------
# Label Encoder
# ----------------------------------------
label_encoder = LabelEncoder()

if os.path.exists(LABEL_CLASSES_PATH):
    label_encoder.classes_ = np.load(LABEL_CLASSES_PATH, allow_pickle=True)
else:
    print(f"Warning: {LABEL_CLASSES_PATH} not found. Creating default classes.")
    default_labels = ["positive", "negative", "neutral"]  # Replace with your training labels
    label_encoder.fit(default_labels)
    np.save(LABEL_CLASSES_PATH, label_encoder.classes_)

# ----------------------------------------
# Prediction Function
# ----------------------------------------
def predict_sentiment(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    pred = MODEL.predict(padded, verbose=0)
    class_idx = np.argmax(pred)
    sentiment = label_encoder.inverse_transform([class_idx])[0]
    confidence = float(pred[0][class_idx])
    return sentiment, confidence

# ----------------------------------------
# Web Routes
# ----------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        sentence = data.get("text", "")
        if not sentence.strip():
            return jsonify({"error": "Empty input"}), 400

        sentiment, confidence = predict_sentiment(sentence)
        return jsonify({"sentiment": sentiment, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------
# Run the Application
# ----------------------------------------
if __name__ == "__main__":
    print("Server is running on http://127.0.0.1:5000")
    app.run(debug=True)

