import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Path to your trained model
MODEL_PATH = 'trained_model.h5'

# Load the trained model
model = load_model(MODEL_PATH)

def preprocess_image(img_path):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img.astype('float32') / 255.0  # Normalize image to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(img_path):
    # Preprocess the image
    img = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(img)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    
    # Return the class label
    return predicted_class[0]
