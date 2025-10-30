"""
FIXED TENSORFLOW SCRIPT
All bugs from the original script have been identified and fixed.
Each fix is documented with explanations.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

print("="*80)
print("FIXED TENSORFLOW SCRIPT - MNIST CLASSIFIER")
print("="*80)

# Load MNIST dataset
print("\n📥 Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"Original training data shape: {x_train.shape}")
print(f"Original test data shape: {x_test.shape}")
print(f"Original training labels shape: {y_train.shape}")
print(f"Label range: {y_train.min()} to {y_train.max()}")

# FIX 1: Normalize pixel values to [0, 1] range
print("\n🔧 FIX 1: Normalizing data to [0, 1] range...")
print("   Original range: [0, 255]")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print(f"   Fixed range: [{x_train.min():.2f}, {x_train.max():.2f}]")

# FIX 2: Reshape data from (samples, 28, 28) to (samples, 784) for Dense layers
print("\n🔧 FIX 2: Reshaping data for Dense layers...")
print(f"   Original shape: {x_train.shape}")
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
print(f"   Fixed shape: {x_train.shape}")

# FIX 3: Convert labels to categorical (one-hot encoding)
print("\n🔧 FIX 3: Converting labels to one-hot encoding...")
print(f"   Original labels shape: {y_train.shape}")
print(f"   Sample original labels: {y_train[:5]}")
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)
print(f"   Fixed labels shape: {y_train_categorical.shape}")
print(f"   Sample one-hot encoded label:\n   {y_train_categorical[0]}")

# Build model
print("\n🏗️  Building model...")
model = keras.Sequential([
    # FIX 4: Correct input shape (784,) matches reshaped data
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    # FIX 5: Correct number of output neurons (10 for MNIST digits 0-9)
    layers.Dense(10, activation='softmax')
])

print("\n🔧 FIX 4: Input shape now matches data shape (784,)")
print("🔧 FIX 5: Output layer has 10 neurons for 10 classes")

# FIX 6: Use correct loss function for multi-class classification
# FIX 7: Use appropriate learning rate
print("\n🔧 FIX 6: Using categorical_crossentropy for multi-class classification")
print("🔧 FIX 7: Using appropriate learning rate (0.001)")

optimizer = keras.optimizers.Adam(learning_rate=0.001)  # Reasonable learning rate
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',  # Correct loss function
    metrics=['accuracy']
)

# Display model summary
print("\n📋 Model Summary:")
model.summary()

# FIX 8: Use appropriate batch size
print("\n🔧 FIX 8: Using appropriate batch size (128)")
print(f"   Dataset size: {len(x_train)}")
print(f"   Batch size: 128")
print(f"   Batches per epoch: {len(x_train) // 128}")

# Train model
print("\n🎯 Training model...")
history = model.fit(
    x_train, y_train_categorical,
    batch_size=128,  # Appropriate batch size
    epochs=5,
    validation_split=0.2,
    verbose=1
)

# FIX 9: Evaluate with properly preprocessed data
print("\n📊 Evaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test_categorical, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# FIX 10: Predictions now have correct shape (samples, 10)
print("\n🔮 Making predictions...")
predictions = model.predict(x_test[:5], verbose=0)
print(f"Predictions shape: {predictions.shape}")
print(f"\nSample predictions (probabilities for each class):")
for i in range(5):
    predicted_class = np.argmax(predictions[i])
    actual_class = y_test[i]
    confidence = predictions[i][predicted_class]
    print(f"   Sample {i}: Predicted={predicted_class}, Actual={actual_class}, "
          f"Confidence={confidence:.2%}")

# Visualize training history
print("\n📈 Training History:")
print(f"   Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"   Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"   Final training loss: {history.history['loss'][-1]:.4f}")
print(f"   Final validation loss: {history.history['val_loss'][-1]:.4f}")

# Additional validation
print("\n✅ All Fixes Applied:")
print("   ✓ Data normalized to [0, 1]")
print("   ✓ Data reshaped to (samples, 784)")
print("   ✓ Labels converted to one-hot encoding")
print("   ✓ Input shape matches data shape")
print("   ✓ Output layer has 10 neurons")
print("   ✓ Using categorical_crossentropy loss")
print("   ✓ Using appropriate learning rate (0.001)")
print("   ✓ Using appropriate batch size (128)")
print("   ✓ Proper data preprocessing for evaluation")
print("   ✓ Predictions have correct shape")

print("\n" + "="*80)
print("✅ SCRIPT COMPLETED SUCCESSFULLY (ALL BUGS FIXED)")
print("="*80)
