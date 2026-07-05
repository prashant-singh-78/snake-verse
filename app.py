import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'snake_classifier_model.keras')
CLASSES_PATH = os.path.join(BASE_DIR, 'classes.txt')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (224, 224)

# Load model and classes
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

def load_classes():
    if not os.path.exists(CLASSES_PATH):
        return ["Non Venomous", "Venomous"]
    with open(CLASSES_PATH, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

class_names = load_classes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Predict
            img = image.load_img(filepath, target_size=IMG_SIZE)
            img_array = image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array)

            if len(class_names) == 2:
                score = predictions[0][0]
                predicted_class = class_names[1] if score > 0.5 else class_names[0]
                confidence = float(score if score > 0.5 else 1 - score) * 100
            else:
                score = tf.nn.softmax(predictions[0])
                predicted_class = class_names[np.argmax(score)]
                confidence = float(np.max(score)) * 100
            
            # Format confidence
            confidence_str = f"{confidence:.2f}%"
            
            # Keep the image URL for the frontend
            img_url = f"/static/uploads/{filename}"
            
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': confidence_str,
                'image_url': img_url
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
