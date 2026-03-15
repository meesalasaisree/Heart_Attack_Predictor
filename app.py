# ======================================================================
# Heart Attack Risk Prediction from Retinal Images
# FILE: app.py (Web Application Backend)
# ======================================================================

from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# --- Configuration ---
IMG_SIZE = (150, 150)
MODEL_PATH = 'model.h5'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the class names
CLASS_NAMES = ['Low_Risk (0)', 'High_Risk (1)']

# Load the trained model once when the application starts
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ----------------------------------------------------------------------
# Helper Function for Image Processing and Prediction
# ----------------------------------------------------------------------
def preprocess_and_predict(image_path):
    """Loads, resizes, and predicts the class of the retinal image."""
    if model is None:
        return "Model Error", 0.0

    try:
        # Load the image
        img = Image.open(image_path).convert('RGB')
        # Resize to the size the model expects
        img = img.resize(IMG_SIZE)
        # Convert image to a numpy array
        img_array = np.array(img, dtype='float32')
        # Normalize the pixel values (just like we did in training)
        img_array = img_array / 255.0
        # Expand dimensions to create a batch (1, 150, 150, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        # Get the predicted class index (0 or 1)
        predicted_class_index = np.argmax(score)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(score) * 100

        return predicted_class_name, confidence

    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Processing Error", 0.0

# ----------------------------------------------------------------------
# Flask Routes (Web Pages)
# ----------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page route to handle GET (display page) and POST (form submit)."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the file securely
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Get the prediction
            prediction, confidence = preprocess_and_predict(file_path)
            
            # Pass results to the prediction page
            return render_template(
                'prediction.html', 
                prediction=prediction, 
                confidence=f"{confidence:.2f}",
                image_file=url_for('static', filename=f'uploads/{filename}')
            )
            
    return render_template('index.html')

if __name__ == '__main__':
    # Run the web application
    print(f"Running Flask app. Visit http://127.0.0.1:5000/")
    app.run(debug=True)