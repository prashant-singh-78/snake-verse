import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'snake_classifier_model.keras')
CLASSES_PATH = os.path.join(BASE_DIR, 'classes.txt')

IMG_SIZE = (224, 224)

def load_classes():
    if not os.path.exists(CLASSES_PATH):
        # Default classes if classes.txt is missing
        return ["Non Venomous", "Venomous"]
    with open(CLASSES_PATH, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def predict_image(img_path, model, class_names):
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    
    if len(class_names) == 2:
        # Binary classification
        score = predictions[0][0]
        # score closer to 0 is class 0, closer to 1 is class 1
        predicted_class = class_names[1] if score > 0.5 else class_names[0]
        confidence = score if score > 0.5 else 1 - score
    else:
        # Multi-class classification
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

    print(f"\n--- Prediction Results ---")
    print(f"Image: {img_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"--------------------------\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run train.py first.")
        sys.exit(1)
        
    print("Loading model...")
    # Suppress TF logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_classes()
    
    predict_image(img_path, model, class_names)
