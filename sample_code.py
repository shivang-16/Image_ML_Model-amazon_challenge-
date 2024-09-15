import os
import pandas as pd
from src.utils import download_images
from src.helper import predict_image

if __name__ == "__main__":
    DATASET_FOLDER = './dataset/'
    IMAGE_FOLDER = './images/'
    
    # Load test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

    # Ensure image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    # Download all images
    print("Downloading images...")
    download_images(test['image_link'], IMAGE_FOLDER)
    
    # Predict values
    print("Making predictions...")
    test['prediction'] = test['image_link'].apply(lambda img_link: predict_image(os.path.join(IMAGE_FOLDER, os.path.basename(img_link))))
    
    # Save predictions to CSV
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")
