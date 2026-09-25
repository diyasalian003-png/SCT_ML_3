"""
Test the trained SVM on any single image you provide.

Usage:
    python predict_image.py path/to/your/photo.jpg
"""

import sys
import numpy as np
from PIL import Image
from skimage.feature import hog
import joblib

IMG_SIZE = (64, 64)

def predict(image_path):
    model = joblib.load("outputs/svm_model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")

    img = Image.open(image_path).convert("L")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0

    feat = hog(
        arr, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys"
    )
    feat_scaled = scaler.transform([feat])

    pred = model.predict(feat_scaled)[0]
    label = "Dog" if pred == 1 else "Cat"

    # Confidence via decision function distance
    decision = model.decision_function(feat_scaled)[0]
    confidence = abs(decision)

    print(f"\nImage: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence score: {confidence:.2f} (higher = more confident)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py path/to/your/photo.jpg")
    else:
        predict(sys.argv[1])
