"""
BUGGY TENSORFLOW SCRIPT
This script contains multiple common TensorFlow errors:
1. Dimension mismatches
2. Incorrect loss functions
3. Wrong optimizer configuration
4. Data preprocessing issues
5. Model architecture errors

Task: Debug and fix all errors
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

print("="*80)
print("BUGGY TENSORFLOW SCRIPT - MNIST CLASSIFIER")
print("="*80)

# Load MNIST dataset
print("\n📥 Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# BUG 1: Incorrect data preprocessing - not normalizing
print("📊 Preprocessing data...")
x_train = x_train  # Missing normalization!
x_test = x_test    # Missing normalization!

# BUG 2: Wrong shape - not reshaping for dense layers
# Should be (60000, 784) but keeping as (60000, 28, 28)
print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# BUG 3: Not converting labels to categorical
# y_train and y_test are integers, not one-hot encoded
print(f"Training labels shape: {y_train.shape}")

# Build model
print("\n🏗️  Building model...")
model = keras.Sequential([
    # BUG 4: Input shape mismatch - expecting flattened input but data is 2D
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    
    # BUG 5: Wrong number of output neurons - should be 10 for MNIST
    layers.Dense(5, activation='softmax')  # Only 5 outputs instead of 10!
])

# BUG 6: Wrong loss function for multi-class classification
# Using binary_crossentropy instead of categorical_crossentropy
print("\n⚙️  Compiling model...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Wrong loss function!
    metrics=['accuracy']
)

# BUG 7: Wrong optimizer learning rate - too high
# Should use a lower learning rate
optimizer = keras.optimizers.Adam(learning_rate=10.0)  # Way too high!
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model summary
print("\n📋 Model Summary:")
model.summary()

# BUG 8: Batch size larger than dataset
# Training with incorrect batch size
print("\n🎯 Training model...")
history = model.fit(
    x_train, y_train,
    batch_size=100000,  # Larger than dataset!
    epochs=5,
    validation_split=0.2,
    verbose=1
)

# BUG 9: Evaluating without proper preprocessing
print("\n📊 Evaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# BUG 10: Prediction shape mismatch
print("\n🔮 Making predictions...")
predictions = model.predict(x_test[:5])
print(f"Predictions shape: {predictions.shape}")
print(f"Sample predictions:\n{predictions}")

print("\n" + "="*80)
print("❌ SCRIPT COMPLETED (WITH BUGS)")
print("="*80)
