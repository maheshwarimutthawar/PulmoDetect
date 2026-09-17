import os
import zipfile
import urllib.request
import tensorflow as tf
from tensorflow.keras import layers, models

print("=== PulmoDetect Training Pipeline Initialized ===")

# 1. Automatic Dataset Downloader
# Yeh URL runtime par ek lightweight sample dataset download karega
DATASET_URL = "https://raw.githubusercontent.com/masterflanker/sample-datasets/main/sample_chest_xray.zip" 
zip_path = "dataset.zip"
extract_path = "dataset"

if not os.path.exists(extract_path):
    print("Downloading dataset...")
    try:
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset ready successfully!")
    except Exception as e:
        print(f"Note: Using fallback synthetic structure due to network/URL: {e}")
        os.makedirs(os.path.join(extract_path, "train", "NORMAL"), exist_ok=True)
        os.makedirs(os.path.join(extract_path, "train", "PNEUMONIA"), exist_ok=True)
else:
    print("Dataset already exists.")

# 2. Define Model Architecture for X-Ray Binary Classification
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),
    
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') # Binary: Normal vs Disease
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
print("=== Model Training Setup Completed Successfully! ===")
