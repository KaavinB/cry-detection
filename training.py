import os
import sys
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Print Python executable path
print(f"Python executable: {sys.executable}")

# Define paths and initialize lists
data_dir = 'C:/Users/kaavi/OneDrive/Desktop/Darshan/Dataset1'
labels = []
features = []

def extract_features(file_path, n_mfcc=40):
    """Extract MFCC features from an audio file."""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
    return None

# Loop through the dataset and extract features
for label in ['cry', 'not_cry']:
    folder_path = os.path.join(data_dir, label)
    print(f"Processing folder: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav') or file_name.endswith('.ogg'):
            file_path = os.path.join(folder_path, file_name)
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(1 if label == 'cry' else 0)
            else:
                print(f"Failed to extract features from {file_path}")

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

print(f"Number of features extracted: {features.shape[0]}")
print(f"Number of labels collected: {labels.shape[0]}")

if len(features) == 0:
    print("No data available for training. Please check the dataset path and content.")
else:
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    print("Data successfully split into training and testing sets.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Create and compile the model
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(40,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the model
    model.save('cry_detection_model.h5')
    print("Model saved as 'cry_detection_model.h5'")

    # Save the feature extraction parameters
    joblib.dump({'n_mfcc': 40}, 'feature_params.joblib')
    print("Feature extraction parameters saved as 'feature_params.joblib'")

print("Training complete. You can now use 'test_model.py' to test the model with your own audio files.")