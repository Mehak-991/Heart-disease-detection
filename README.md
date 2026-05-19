# Heart Disease Detection Web App

This project is a complete prediction and deployment pipeline to detect heart disease using two inference modes:

- Image-based: a Convolutional Neural Network (CNN) that accepts MRI/heart images.
- Tabular-based: a Random Forest classifier trained on clinical features (age, sex, blood pressure, cholesterol, etc.).

🚀 Project Highlights

- Dataset: provided in `data/heart.csv` (preprocessed in notebooks under `model/`).
- Models: `model/cnn_model.h5` (CNN) and optional `model/random_forest_model.pkl` (Random Forest).
- Deployment: Flask web interface (`app.py`) with HTML forms for users.
- Output: prediction message (disease / no disease) and model probability when available.

📁 Dataset

The dataset `data/heart.csv` contains patient records with features such as:

- age, sex, cp (chest pain), trestbps (resting blood pressure), chol (cholesterol)
- fbs, restecg, thalach (max heart rate), exang, oldpeak, slope, ca, thal

Data preprocessing (handled in notebooks) includes cleaning, feature conversions, and optional feature engineering.

🧠 Model Building (training notebooks)

- Training artifacts and example notebooks are under the `model/` directory. Notebooks demonstrate data cleaning, model training and evaluation.
- Typical training flow for the Random Forest (if you retrain):
	- Clean dataset and encode categorical fields
	- Split into train/test sets
	- Optionally apply resampling (SMOTE/SMOTEENN) to address class imbalance
	- Train RandomForestClassifier (e.g., `n_estimators=100, max_depth=6, min_samples_leaf=8`)
	- Evaluate with accuracy, classification report, and confusion matrix
	- Export final model using `pickle` → `model/random_forest_model.pkl`

🌐 Deployment (`app.py`)

- Built with Flask. Routes include:
	- `/` → Home page (`templates/index.html`)
	- `/form` → Tabular input form (`templates/form.html`)
	- `/predict` → `POST` endpoint: accepts either an uploaded image (`mri_image`) or tabular form fields. The app:
		- Uses the CNN model for image files (resizes to 224×224 and normalizes inputs).
		- Uses the Random Forest model for tabular input (expects 13 features in the same order used during training).

🧾 Example Output

✅ This patient is likely to continue healthy!

🔒 Confidence: 85.27% (when probability is available)

🛠 Requirements

All dependencies are listed in `requirements.txt`.

Install required packages:

```bash
pip install -r requirements.txt
```

▶️ Running the App

Place your model files in the `model/` directory (`cnn_model.h5`, optional `random_forest_model.pkl`) and run:

```bash
python app.py
```

Open http://localhost:5000 in your browser. To run on a custom port or in a container, set `PORT` before starting the app. When `PORT` is set the app binds to `0.0.0.0`.

📁 Project Structure

```
├── app.py                  # Flask web app
├── model/                  # Trained model artifacts and notebooks
│   ├── cnn_model.h5
│   └── random_forest_model.pkl (optional)
├── data/                   # Datasets (heart.csv)
├── templates/              # HTML templates (index.html, form.html, result.html)
├── static/                 # CSS and images
├── uploads/                # Uploaded files (runtime)
├── requirements.txt
├── Dockerfile
└── README.md
```

📌 Notes

- The Random Forest model is optional. If `model/random_forest_model.pkl` is missing, the tabular form will return an informative error message.
- The CNN expects 224×224 RGB images and a sigmoid output producing a disease probability.

🧪 Technologies Used

- Python 3.8+
- Flask
- TensorFlow / Keras (for CNN inference)
- scikit-learn (Random Forest)
- NumPy, Pandas, Pillow

📦 Docker

Build and run using Docker:

```bash
docker build -t heart-disease-app .
docker run -p 5000:5000 -e PORT=5000 heart-disease-app
```

🔧 Troubleshooting

- Missing Random Forest: add `random_forest_model.pkl` to `model/`.
- TensorFlow logs: `app.py` suppresses many TF logs; adjust environment variables in `app.py` if you need GPU support.
- Remove `debug=True` in production to avoid exposing debug information.

🤝 Contributing

Contributions welcome — open an issue or submit a PR. Include tests or notebooks for retraining if applicable.

## License

MIT

