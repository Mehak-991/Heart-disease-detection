import os
# Suppress TensorFlow warnings/info logs BEFORE importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # Hide INFO/WARNING/ERROR from TF C++
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # Disable oneDNN warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'           # Force CPU (skip GPU probe)

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import pickle
import logging

# Suppress TF Python-level logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

app = Flask(__name__)

# ---------------- Load Models (lazy) ---------------- #
CNN_MODEL_PATH = os.path.join("model", "cnn_model.h5")
RF_MODEL_PATH = os.path.join("model", "random_forest_model.pkl")

cnn_model = None
rf_model = None

def get_cnn_model():
    global cnn_model
    if cnn_model is None:
        cnn_model = load_model(CNN_MODEL_PATH)
    return cnn_model

def get_rf_model():
    global rf_model
    if rf_model is None and os.path.exists(RF_MODEL_PATH):
        with open(RF_MODEL_PATH, "rb") as f:
            rf_model = pickle.load(f)
    return rf_model

# ---------------- Routes ---------------- #

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Form page
@app.route('/form')
def form():
    return render_template("form.html")

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('mri_image')  # may be None if no file uploaded

        # ✅ CASE 1: MRI image uploaded → CNN model
        if file and file.filename != '':
            img = Image.open(file.stream).convert("RGB")
            img = img.resize((224, 224))

            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            model = get_cnn_model()
            prediction = model.predict(img_array)
            prob = prediction[0][0]
            result = 1 if prob > 0.5 else 0

            message = "Congrats! No heart disease detected." if result == 0 else "Heart disease detected."

            return render_template("result.html", prediction=message)

        # ✅ CASE 2: Tabular input → RF model
        else:
            model = get_rf_model()
            if model is None:
                return render_template("result.html",
                                       prediction="Error: Random Forest model not found on server.")

            # Collect form values
            features = [
                float(request.form.get("age", 0)),
                float(request.form.get("sex", 0)),
                float(request.form.get("cp", 0)),
                float(request.form.get("trestbps", 0)),
                float(request.form.get("chol", 0)),
                float(request.form.get("fbs", 0)),
                float(request.form.get("restecg", 0)),
                float(request.form.get("thalach", 0)),
                float(request.form.get("exang", 0)),
                float(request.form.get("oldpeak", 0)),
                float(request.form.get("slope", 0)),
                float(request.form.get("ca", 0)),
                float(request.form.get("thal", 0))
            ]

            features = np.array(features).reshape(1, -1)

            pred = model.predict(features)[0]
            message = "Congrats! No heart disease detected." if pred == 0 else "Heart disease detected."

            return render_template("result.html", prediction=message)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")


# ---------------- Run ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    app.run(host=host, port=port, debug=True)
