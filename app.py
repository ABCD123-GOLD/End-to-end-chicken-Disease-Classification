from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
import sys

# --- Import the PredictionPipeline from your predict.py file ---
from Chicken_Disease_Classification.pipeline.predict import PredictionPipeline

# Initialize the Flask app
app = Flask(__name__)

# --- Load the model once on startup ---
# FINAL CORRECTION: Using the .keras file extension as confirmed.
MODEL_PATH = os.path.join("artifacts", "training", "trained_model_best.keras")

try:
    # This 'model' object will be passed to your pipeline
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure the model file exists at the specified path.")
    sys.exit(1)

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main HTML page for the user."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    """Handles the image upload and returns the prediction as JSON."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    f = request.files['image']
    
    # Create a secure, temporary path to save the uploaded file
    basepath = os.path.dirname(__file__)
    uploads_path = os.path.join(basepath, 'uploads')
    os.makedirs(uploads_path, exist_ok=True) # Ensure the directory exists
    
    file_path = os.path.join(uploads_path, secure_filename(f.filename))
    f.save(file_path)

    try:
        # 1. Create an instance of your pipeline with the file path
        pipeline = PredictionPipeline(file_path)
        
        # 2. Call the predict method, passing the globally loaded model
        prediction_result = pipeline.predict(model)
        
        # 3. Return the result to the frontend
        return jsonify({'prediction': prediction_result})
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Failed to process the image.'}), 500
    finally:
        # Clean up by removing the uploaded file after prediction
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    # Set debug=False for production
    app.run(debug=True)

