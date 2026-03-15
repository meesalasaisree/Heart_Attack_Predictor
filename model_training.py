# ======================================================================
# Heart Attack Risk Prediction from Retinal Images
# FILE: model_training.py (Optimized Custom CNN with Class Weighting)
# ======================================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight # We will need this for weighting

# --- Configuration ---
BASE_DIR = 'data' 
IMG_SIZE = (150, 150) 
NUM_CLASSES = 2 
BATCH_SIZE = 32 
EPOCHS = 30 # Increased epochs for better learning

def build_model():
    """Defines the Optimized Custom Convolutional Neural Network (CNN) architecture."""
    model = Sequential([
        # Convolutional Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2, 2),
        
        # Convolutional Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Convolutional Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Convolutional Block 4
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Flatten 
        Flatten(),
        
        # Fully Connected Layer 
        Dense(1024, activation='relu'),
        
        # Output Layer 
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

def setup_data_generators():
    """Prepares data from folders using Keras's ImageDataGenerator with Augmentation."""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Keras automatically assigns class_indices based on folder names
    train_generator = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_test_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'validation'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = validation_test_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

if __name__ == '__main__':
    
    if not os.path.isdir(os.path.join(BASE_DIR, 'train', 'Low_Risk')):
        print("ERROR: Data structure not found!")
        exit()

    print("--- Starting Optimized Custom CNN Setup ---")
    
    train_gen, validation_gen, test_gen = setup_data_generators()
    
    # ==================================================================
    # STEP 2: CALCULATE AND APPLY CLASS WEIGHTS
    # ==================================================================
    # 1. Get the actual class indices and counts from the generator
    # .classes is the correct attribute for counts (0, 1, 0, 1, 1...)
    class_labels = train_gen.classes 
    class_names = list(train_gen.class_indices.keys())
    
    # 2. Compute class weights to inverse balance the classes
    # (e.g., if class 0 is 3x larger than class 1, class 1 gets 3x the weight)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(class_labels),
        y=class_labels
    )
    
    # 3. Convert weights to a dictionary format Keras expects
    class_weights = dict(zip(np.unique(class_labels), weights))

    print("\n--- CLASS WEIGHTING APPLIED ---")
    print(f"Class Names: {train_gen.class_indices}")
    print(f"Calculated Weights: {class_weights}")
    print("--------------------------------\n")
    # ==================================================================

    model = build_model()
    model.summary() 
    
    print(f"\n--- Starting Model Training ({EPOCHS} Epochs) with WEIGHTS ---")

    # Train the model, passing the class_weights dictionary
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_gen,
        validation_steps=validation_gen.samples // BATCH_SIZE,
        # THE FIX: Apply the calculated class weights here!
        class_weight=class_weights 
    )

    model_filename = 'model.h5'
    model.save(model_filename) 
    print(f"\nModel training complete. Final model saved as '{model_filename}'")
    
    print("\n--- Evaluating Model on Test Data ---")
    loss, accuracy = model.evaluate(test_gen, steps=test_gen.samples // BATCH_SIZE)
    print(f"Final Test Loss: {loss:.4f}")
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")