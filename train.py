"""
Emotion Recognition Model Training Script
Dataset: FER2013 (Facial Expression Recognition)
Model: CNN with 3 convolutional layers
Training: 120 epochs, batch size 64
"""

import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================
# 1. CONFIGURATION
# ============================================

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Directories
TRAIN_DIR = 'train'
VAL_DIR = 'test'
DATASET_CSV = 'fer2013.csv'  # Download from Kaggle

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 120
IMAGE_SIZE = 48

# Emotion labels
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# ============================================
# 2. UTILITY FUNCTIONS
# ============================================

def atoi(s):
    """Convert string to integer (custom implementation)"""
    n = 0
    for i in s:
        n = n * 10 + ord(i) - ord("0")
    return n

def create_folders():
    """Create train/test folders for each emotion"""
    outer_names = ['train', 'test']
    for outer_name in outer_names:
        for inner_name in EMOTIONS:
            os.makedirs(os.path.join(outer_name, inner_name), exist_ok=True)
    print("📁 Folders created successfully!")

def plot_model_history(model_history, save_path='plot.png'):
    """
    Plot Accuracy and Loss curves from training history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), 
                model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), 
                model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')
    
    # Loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), 
                model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), 
                model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    
    fig.savefig(save_path)
    print(f"📊 Plot saved to {save_path}")
    plt.show()

# ============================================
# 3. LOAD AND PREPARE DATASET
# ============================================

def load_dataset(csv_path):
    """Load FER2013 dataset from CSV"""
    df = pd.read_csv(csv_path)
    print(f"📊 Dataset loaded: {len(df)} samples")
    return df

def extract_images(df, train_dir='train', test_dir='test'):
    """
    Extract images from CSV and save to train/test folders
    """
    # Create directories
    create_folders()
    
    # Initialize counters
    counters = {
        'train': {emotion: 0 for emotion in EMOTIONS},
        'test': {emotion: 0 for emotion in EMOTIONS}
    }
    
    mat = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    
    print("🖼️ Extracting images...")
    
    for i in tqdm(range(len(df))):
        txt = df['pixels'][i]
        words = txt.split()
        
        # Convert pixels to image
        for j in range(IMAGE_SIZE * IMAGE_SIZE):
            xind = j // IMAGE_SIZE
            yind = j % IMAGE_SIZE
            mat[xind][yind] = atoi(words[j])
        
        img = Image.fromarray(mat)
        emotion_idx = df['emotion'][i]
        emotion_name = EMOTIONS[emotion_idx]
        
        # Determine train or test
        if i < 28709:  # First 28,709 samples are training
            folder = 'train'
        else:
            folder = 'test'
        
        # Save image
        filename = f"{folder}/{emotion_name}/im{counters[folder][emotion_name]}.png"
        img.save(filename)
        counters[folder][emotion_name] += 1
    
    print("✅ Image extraction complete!")
    print(f"   Train samples: {sum(counters['train'].values())}")
    print(f"   Test samples: {sum(counters['test'].values())}")

# ============================================
# 4. BUILD MODEL
# ============================================

def build_model():
    """Build CNN model for emotion recognition"""
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                     input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    # Compile
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    print("🏗️ Model built successfully!")
    model.summary()
    return model

# ============================================
# 5. MAIN TRAINING PIPELINE
# ============================================

def main():
    """
    Main training pipeline
    Steps:
    1. Load FER2013 dataset
    2. Extract images to folders
    3. Build CNN model
    4. Train the model
    5. Save trained model
    """
    
    # Step 1: Load dataset
    print("=" * 50)
    print("🧠 EMOTION RECOGNITION TRAINING")
    print("=" * 50)
    
    # Make sure you have fer2013.csv in the current directory
    # Download from: https://www.kaggle.com/datasets/deadskull7/fer2013
    if not os.path.exists(DATASET_CSV):
        print(f"⚠️  Error: {DATASET_CSV} not found!")
        print("Please download fer2013.csv from Kaggle:")
        print("https://www.kaggle.com/datasets/deadskull7/fer2013")
        return
    
    df = load_dataset(DATASET_CSV)
    
    # Step 2: Extract images (optional - skip if already extracted)
    if not os.path.exists('train') or len(os.listdir('train')) == 0:
        extract_images(df)
    else:
        print("📁 Images already extracted, skipping...")
    
    # Step 3: Data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical'
    )
    
    # Step 4: Build model
    model = build_model()
    
    # Step 5: Train
    print(f"🚀 Starting training for {NUM_EPOCHS} epochs...")
    
    model_info = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )
    
    # Step 6: Save model
    model.save('model.h5')
    print("✅ Model saved as 'model.h5'")
    
    # Step 7: Plot results
    plot_model_history(model_info)
    
    print("\n🎉 Training complete!")
    print("Use 'model.h5' in your emotion detection application.")

# ============================================
# 6. RUN
# ============================================

if __name__ == "__main__":
    main()