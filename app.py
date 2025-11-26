import os
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from model import load_trained_model, predict_ecg
from utils import preprocess_image

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_demo_only'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
model = None
class_mapping = {}

def init_model():
    global model, class_mapping
    try:
        model_data = load_trained_model()
        if model_data:
            print("Model loaded successfully.")
            model = model_data
            if 'class_names' in model_data:
                class_mapping = model_data['class_names']
        else:
            print("Model not found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze immediately
        return redirect(url_for('analyze', filename=filename))

@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not model:
        # Try to initialize again just in case
        init_model()
        
    if not model:
        flash("Model is not trained or loaded. Please run training script.")
        return redirect(url_for('index'))
        
    # Preprocess
    img = preprocess_image(filepath)
    if img is None:
        flash("Error processing image. Please upload a valid image.")
        return redirect(url_for('index'))
    
    # Predict
    label, confidence = predict_ecg(model, img, class_mapping)
    
    # Convert confidence to percentage
    accuracy = f"{confidence * 100:.2f}%"
    
    return render_template('result.html', 
                           filename=filename, 
                           label=label, 
                           accuracy=accuracy,
                           confidence=confidence) # Pass raw confidence for charts if needed

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

if __name__ == '__main__':
    init_model()
    app.run(debug=True, port=5000)
