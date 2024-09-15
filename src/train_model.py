import os
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image as PILImage
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# Constants
IMAGE_FOLDER = './images/'
DATASET_FOLDER = './dataset/'
TRAIN_CSV = os.path.join(DATASET_FOLDER, 'train.csv')
MODEL_SAVE_PATH = 'trained_model.h5'
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

# Ensure image folder exists
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

def download_image(img_url, local_path):
    try:
        response = requests.get(img_url)
        response.raise_for_status()  # Check for HTTP errors
        image = PILImage.open(BytesIO(response.content))
        image.save(local_path)
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")

def load_image(img_path):
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img.astype('float32') / 255.0  # Normalize image to [0, 1]
        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))  # Return a blank image if there's an error

def load_data():
    df = pd.read_csv(TRAIN_CSV)
    images = df['image_link'].values
    labels = df[['entity_name', 'entity_value']].apply(lambda x: f"{x['entity_value']} {x['entity_name']}", axis=1).values
    
    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    
    return images, labels, le

def create_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_tf_dataset(images, labels, batch_size=BATCH_SIZE):
    def load_and_preprocess_image(img_path):
        local_path = os.path.join(IMAGE_FOLDER, os.path.basename(img_path))
        if not os.path.exists(local_path):
            download_image(img_path, local_path)
        img = load_image(local_path)
        return img

    def gen():
        for img_path, label in zip(images, labels):
            yield load_and_preprocess_image(img_path), label

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def main():
    images, labels, label_encoder = load_data()
    
    # Create model
    num_classes = len(label_encoder.classes_)
    model = create_model(num_classes)
    
    # Create dataset
    dataset = create_tf_dataset(images, labels)
    
    # Train model
    print("Training model...")
    model.fit(
        dataset,
        steps_per_epoch=len(images) // BATCH_SIZE,
        epochs=10  # Set the number of epochs
    )
    
    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
