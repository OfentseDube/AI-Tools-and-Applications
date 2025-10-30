"""
MNIST Digit Classifier - Streamlit Web Application
A professional web interface for handwritten digit recognition
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import cv2
import io
import base64
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-number {
        font-size: 5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .confidence-text {
        font-size: 1.5rem;
        color: #555;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔢 MNIST Digit Classifier</h1>', unsafe_allow_html=True)
st.markdown("### Draw a digit or upload an image to get predictions!")

# Load or create model
@st.cache_resource
def load_model():
    """Load or train the MNIST model"""
    try:
        # Try to load existing model
        model = keras.models.load_model('mnist_model.h5')
        st.sidebar.success("✅ Model loaded successfully!")
    except:
        # Train a new model if not found
        st.sidebar.info("🔄 Training new model...")
        
        # Load MNIST data
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Preprocess
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        # Build model
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        with st.spinner("Training model... This may take a minute."):
            model.fit(
                x_train, y_train,
                batch_size=128,
                epochs=5,
                validation_split=0.2,
                verbose=0
            )
        
        # Save model
        model.save('mnist_model.h5')
        st.sidebar.success("✅ Model trained and saved!")
    
    return model

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    
    # Invert if needed (MNIST has white digits on black background)
    if np.mean(image) > 127:
        image = 255 - image
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Reshape for model
    image = image.reshape(1, 784)
    
    return image

def predict_digit(model, image):
    """Make prediction on preprocessed image"""
    prediction = model.predict(image, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    return predicted_class, confidence, prediction[0]

# Load model
model = load_model()

# Sidebar
st.sidebar.title("📊 Model Information")
st.sidebar.markdown("""
**Architecture:**
- Input Layer: 784 neurons (28×28 pixels)
- Hidden Layer 1: 128 neurons (ReLU)
- Dropout: 0.2
- Hidden Layer 2: 64 neurons (ReLU)
- Dropout: 0.2
- Output Layer: 10 neurons (Softmax)

**Training:**
- Dataset: MNIST (60,000 training images)
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Accuracy: ~97-98%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 How to Use")
st.sidebar.markdown("""
1. **Draw Mode**: Draw a digit on the canvas
2. **Upload Mode**: Upload an image file
3. Click **Predict** to get results
4. View confidence scores for all digits
""")

# Main content - Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🎨 Input Method")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Draw on Canvas", "Upload Image"],
        horizontal=True
    )
    
    input_image = None
    
    if input_method == "Draw on Canvas":
        st.markdown("#### Draw a digit (0-9)")
        
        # Canvas for drawing
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if canvas_result.image_data is not None:
            input_image = canvas_result.image_data
    
    else:  # Upload Image
        st.markdown("#### Upload an image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a handwritten digit"
        )
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            input_image = np.array(image)
            
            # Display uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict button
    predict_button = st.button("🔮 Predict Digit", use_container_width=True)

with col2:
    st.markdown("### 📊 Prediction Results")
    
    if predict_button and input_image is not None:
        # Preprocess image
        processed_image = preprocess_image(input_image)
        
        # Make prediction
        predicted_digit, confidence, all_probabilities = predict_digit(model, processed_image)
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-number">{predicted_digit}</div>
            <div class="confidence-text">Confidence: {confidence:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display preprocessed image
        st.markdown("#### Preprocessed Image (28×28)")
        preprocessed_display = processed_image.reshape(28, 28)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(preprocessed_display, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        
        # Confidence scores for all digits
        st.markdown("#### Confidence Scores for All Digits")
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        digits = list(range(10))
        colors = ['#1f77b4' if i == predicted_digit else '#d3d3d3' for i in digits]
        bars = ax.bar(digits, all_probabilities, color=colors)
        ax.set_xlabel('Digit', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title('Prediction Confidence for Each Digit', fontsize=14)
        ax.set_xticks(digits)
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, prob in zip(bars, all_probabilities):
            height = bar.get_height()
            if height > 0.01:  # Only show labels for visible bars
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.1%}',
                       ha='center', va='bottom', fontsize=9)
        
        st.pyplot(fig)
        
        # Show top 3 predictions
        st.markdown("#### Top 3 Predictions")
        top_3_indices = np.argsort(all_probabilities)[-3:][::-1]
        
        for i, idx in enumerate(top_3_indices, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            st.markdown(f"{medal} **Digit {idx}**: {all_probabilities[idx]:.2%}")
    
    elif predict_button:
        st.warning("⚠️ Please draw a digit or upload an image first!")
    else:
        st.info("👈 Draw a digit or upload an image, then click Predict!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Built with Streamlit & TensorFlow | MNIST Digit Classifier</p>
    <p>Model Accuracy: ~97-98% on test set</p>
</div>
""", unsafe_allow_html=True)

# Additional features in expander
with st.expander("ℹ️ About MNIST Dataset"):
    st.markdown("""
    ### MNIST Dataset
    
    The **MNIST database** (Modified National Institute of Standards and Technology database) 
    is a large database of handwritten digits commonly used for training various image 
    processing systems.
    
    **Dataset Statistics:**
    - Training set: 60,000 images
    - Test set: 10,000 images
    - Image size: 28×28 pixels (grayscale)
    - Classes: 10 (digits 0-9)
    
    **Applications:**
    - Handwriting recognition
    - Optical character recognition (OCR)
    - Automated form processing
    - Check processing
    - Postal code recognition
    """)

with st.expander("🔧 Technical Details"):
    st.markdown("""
    ### Model Architecture
    
    ```python
    model = keras.Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    ```
    
    ### Training Configuration
    - **Optimizer**: Adam (learning_rate=0.001)
    - **Loss Function**: Categorical Crossentropy
    - **Batch Size**: 128
    - **Epochs**: 5
    - **Validation Split**: 20%
    
    ### Preprocessing Steps
    1. Convert to grayscale (if needed)
    2. Resize to 28×28 pixels
    3. Invert colors (white digit on black background)
    4. Normalize pixel values to [0, 1]
    5. Flatten to 784-dimensional vector
    """)

# Sample images section
with st.expander("📸 Try Sample MNIST Images"):
    st.markdown("### Load Sample Images from MNIST Dataset")
    
    if st.button("Load Random Sample"):
        # Load MNIST test data
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Select random samples
        indices = np.random.choice(len(x_test), 5, replace=False)
        
        cols = st.columns(5)
        for i, idx in enumerate(indices):
            with cols[i]:
                sample_image = x_test[idx]
                actual_label = y_test[idx]
                
                # Display image
                st.image(sample_image, caption=f"Actual: {actual_label}", 
                        use_column_width=True, clamp=True)
                
                # Make prediction
                processed = preprocess_image(sample_image)
                pred_digit, conf, _ = predict_digit(model, processed)
                
                st.markdown(f"**Predicted: {pred_digit}**")
                st.markdown(f"Confidence: {conf:.1%}")
