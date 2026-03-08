from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import pickle

app = Flask(__name__)

# ---------------- Load Models ---------------- #
CNN_MODEL_PATH = os.path.join("model", "cnn_model.h5")
RF_MODEL_PATH = os.path.join("model", "random_forest_model.pkl")

# Load CNN model
cnn_model = load_model(CNN_MODEL_PATH)

# Load RF model safely
rf_model = None
if os.path.exists(RF_MODEL_PATH):
    with open(RF_MODEL_PATH, "rb") as f:
        rf_model = pickle.load(f)
else:
    print("⚠️ Random Forest model file not found!")

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

            prediction = cnn_model.predict(img_array)
            prob = prediction[0][0]
            result = 1 if prob > 0.5 else 0

            message = "Congrats! No heart disease detected." if result == 0 else "Heart disease detected."

            return render_template("result.html", prediction=message)

        # ✅ CASE 2: Tabular input → RF model
        else:
            if rf_model is None:
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

            pred = rf_model.predict(features)[0]
            message = "Congrats! No heart disease detected." if pred == 0 else "Heart disease detected."

            return render_template("result.html", prediction=message)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")


# ---------------- Run ---------------- #
if __name__ == "__main__":
    app.run(debug=True)

