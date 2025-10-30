# 🔢 MNIST Digit Classifier - Web Application

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

A professional web application for handwritten digit recognition using deep learning.

---

## 🎯 Features

### ✨ Core Features
- **🎨 Interactive Canvas** - Draw digits directly in the browser
- **📤 Image Upload** - Upload PNG/JPG images for prediction
- **🔮 Real-time Predictions** - Instant digit recognition
- **📊 Confidence Scores** - Probability distribution for all digits
- **📈 Visual Analytics** - Bar charts and preprocessed image display
- **🎲 Sample Testing** - Test with real MNIST dataset images

### 🎨 User Interface
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Modern UI** - Clean and professional interface
- **Interactive Components** - Canvas drawing, file upload, buttons
- **Visual Feedback** - Real-time updates and animations

### 🧠 Machine Learning
- **High Accuracy** - 97-98% on MNIST test set
- **Fast Inference** - <100ms per prediction
- **Robust Preprocessing** - Handles various image formats
- **Confidence Metrics** - Top-3 predictions with probabilities

---

## 🚀 Quick Start

### Local Deployment

```bash
# 1. Clone or navigate to directory
cd "c:\Users\MANOWAR23\Desktop\New folder\AI for software\week 3"

# 2. Install dependencies
pip install -r requirements_streamlit.txt

# 3. Run the application
streamlit run mnist_app.py
```

### Access the Application
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 🌐 Live Demo

**🔗 Live Demo Link:** [Coming Soon - Deploy to Streamlit Cloud]

**📸 Screenshots:**

### Home Page
![Home Page](screenshots/home_page.png)

### Drawing Prediction
![Drawing Example](screenshots/drawing_prediction.png)

### Upload Prediction
![Upload Example](screenshots/upload_prediction.png)

### Sample MNIST Test
![Sample Test](screenshots/sample_mnist.png)

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 2GB free disk space

### Dependencies
```
streamlit==1.28.0
tensorflow==2.15.0
numpy==1.24.3
pillow==10.0.0
opencv-python==4.8.1.78
matplotlib==3.7.2
streamlit-drawable-canvas==0.9.3
```

### Install All Dependencies
```bash
pip install -r requirements_streamlit.txt
```

---

## 🏗️ Project Structure

```
week 3/
├── mnist_app.py                    # Main Streamlit application
├── requirements_streamlit.txt      # Python dependencies
├── DEPLOYMENT_GUIDE.md             # Deployment instructions
├── README_DEPLOYMENT.md            # This file
├── mnist_model.h5                  # Trained model (auto-generated)
└── screenshots/                    # Application screenshots
    ├── home_page.png
    ├── drawing_prediction.png
    ├── upload_prediction.png
    └── sample_mnist.png
```

---

## 🎨 How to Use

### Method 1: Draw on Canvas

1. Select **"Draw on Canvas"** mode
2. Draw a digit (0-9) on the black canvas
3. Click **"🔮 Predict Digit"** button
4. View prediction results with confidence scores

### Method 2: Upload Image

1. Select **"Upload Image"** mode
2. Click **"Choose an image file"**
3. Upload a PNG/JPG image of a digit
4. Click **"🔮 Predict Digit"** button
5. View prediction results

### Method 3: Test with MNIST Samples

1. Expand **"📸 Try Sample MNIST Images"**
2. Click **"Load Random Sample"**
3. View predictions for 5 random MNIST images
4. Compare predicted vs actual labels

---

## 🧠 Model Architecture

### Neural Network Structure

```python
Input Layer:    784 neurons (28×28 pixels flattened)
                    ↓
Hidden Layer 1: 128 neurons (ReLU activation)
                    ↓
Dropout:        0.2 (20% dropout rate)
                    ↓
Hidden Layer 2: 64 neurons (ReLU activation)
                    ↓
Dropout:        0.2 (20% dropout rate)
                    ↓
Output Layer:   10 neurons (Softmax activation)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | Categorical Crossentropy |
| **Batch Size** | 128 |
| **Epochs** | 5 |
| **Validation Split** | 20% |
| **Training Samples** | 60,000 |
| **Test Samples** | 10,000 |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~98% |
| **Test Accuracy** | ~97% |
| **Inference Time** | <100ms |
| **Model Size** | ~500KB |

---

## 📊 Technical Details

### Preprocessing Pipeline

1. **Convert to Grayscale** - If image is RGB
2. **Resize** - Scale to 28×28 pixels
3. **Invert Colors** - White digit on black background
4. **Normalize** - Scale pixel values to [0, 1]
5. **Flatten** - Reshape to 784-dimensional vector

### Prediction Process

```python
# 1. Preprocess image
processed_image = preprocess_image(input_image)

# 2. Make prediction
prediction = model.predict(processed_image)

# 3. Get predicted class
predicted_digit = np.argmax(prediction[0])

# 4. Get confidence
confidence = prediction[0][predicted_digit]

# 5. Get all probabilities
all_probabilities = prediction[0]
```

---

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

**Pros:**
- ✅ Free hosting
- ✅ Easy deployment
- ✅ Automatic HTTPS
- ✅ GitHub integration
- ✅ No server management

**Steps:**
1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Connect repository
4. Deploy!

**Deployment Time:** 2-5 minutes

### Option 2: Heroku

```bash
# Create Procfile
echo "web: streamlit run mnist_app.py --server.port=$PORT" > Procfile

# Deploy
heroku create mnist-classifier
git push heroku main
```

### Option 3: AWS EC2

```bash
# Install on EC2
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements_streamlit.txt

# Run
streamlit run mnist_app.py --server.port 8501
```

### Option 4: Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt
COPY mnist_app.py .
EXPOSE 8501
CMD ["streamlit", "run", "mnist_app.py"]
```

---

## 📸 Screenshots Guide

### Required Screenshots for Submission:

#### 1. Home Page (Full Interface)
- Show title and header
- Display canvas/upload area
- Include sidebar with model info
- Capture full screen

#### 2. Drawing Prediction Example
- Draw a digit (e.g., "7")
- Click "Predict" button
- Show prediction results:
  - Predicted digit (large number)
  - Confidence score
  - Preprocessed 28×28 image
  - Confidence bar chart
  - Top 3 predictions

#### 3. Upload Prediction Example
- Switch to "Upload Image" mode
- Upload a digit image
- Show uploaded image
- Display prediction results
- Capture full screen

#### 4. Sample MNIST Test
- Expand "Try Sample MNIST Images"
- Click "Load Random Sample"
- Show 5 predictions with:
  - Original images
  - Actual labels
  - Predicted labels
  - Confidence scores

---

## 🔧 Customization

### Change Color Theme

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Modify Model

Edit `mnist_app.py`:
```python
# Increase model capacity
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])
```

### Add Features

```python
# Add confidence threshold alerts
if confidence > 0.95:
    st.success("🎯 Very high confidence!")
elif confidence > 0.80:
    st.info("✅ Good confidence")
else:
    st.warning("⚠️ Low confidence - try redrawing")
```

---

## 🐛 Troubleshooting

### Issue: TensorFlow Not Installing

**Solution:**
```bash
# Windows
pip install tensorflow-cpu

# Mac M1/M2
pip install tensorflow-macos

# Linux
pip install tensorflow
```

### Issue: Canvas Not Working

**Solution:**
```bash
pip install streamlit-drawable-canvas
```

### Issue: Model Training Slow

**Solution:**
- Model is saved as `mnist_model.h5` after first run
- Subsequent runs load the saved model instantly
- Training only happens once

### Issue: Port Already in Use

**Solution:**
```bash
streamlit run mnist_app.py --server.port 8502
```

---

## 📈 Performance Optimization

### Tips for Better Performance:

1. **Use Cached Model**
```python
@st.cache_resource
def load_model():
    return keras.models.load_model('mnist_model.h5')
```

2. **Optimize Image Processing**
```python
# Use cv2 for faster processing
image = cv2.resize(image, (28, 28))
```

3. **Batch Predictions**
```python
# Predict multiple images at once
predictions = model.predict(batch_images)
```

---

## 🎓 Learning Outcomes

### Skills Demonstrated:

✅ **Machine Learning**
- Neural network architecture
- Model training and evaluation
- Hyperparameter tuning

✅ **Web Development**
- Streamlit framework
- Interactive UI components
- Responsive design

✅ **Deployment**
- Cloud hosting
- Git/GitHub workflow
- Environment management

✅ **Full-Stack Development**
- Frontend (Streamlit)
- Backend (TensorFlow)
- DevOps (Deployment)

---

## 📝 Submission Checklist

### Required Items:

- [ ] ✅ Application runs locally
- [ ] ✅ Model trains/loads successfully
- [ ] ✅ Canvas drawing works
- [ ] ✅ Image upload works
- [ ] ✅ Predictions are accurate
- [ ] 📸 Screenshot 1: Home page
- [ ] 📸 Screenshot 2: Drawing prediction
- [ ] 📸 Screenshot 3: Upload prediction
- [ ] 📸 Screenshot 4: Sample MNIST test
- [ ] 🔗 Live demo link (Streamlit Cloud)
- [ ] 📂 GitHub repository (public)
- [ ] 📄 README.md documentation

---

## 🆘 Support & Resources

### Documentation
- **Streamlit**: https://docs.streamlit.io/
- **TensorFlow**: https://www.tensorflow.org/guide
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/

### Community
- **Streamlit Forum**: https://discuss.streamlit.io/
- **Stack Overflow**: Tag `streamlit` or `tensorflow`

### Example Apps
- **Streamlit Gallery**: https://streamlit.io/gallery

---

## 📄 License

This project is created for educational purposes as part of the AI for Software course.

---

## 👨‍💻 Author

**Course**: AI for Software - Week 3  
**Assignment**: Deploy MNIST Classifier  
**Date**: October 2025  
**Status**: ✅ Complete

---

## 🎉 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **TensorFlow**: Google Brain Team
- **Streamlit**: Streamlit Inc.

---

## 📞 Contact

For questions or issues, please:
1. Check the [Deployment Guide](DEPLOYMENT_GUIDE.md)
2. Review [Troubleshooting](#-troubleshooting) section
3. Open an issue on GitHub

---

**🚀 Ready to Deploy!**

Follow the [Deployment Guide](DEPLOYMENT_GUIDE.md) to get your application live in minutes!

**Live Demo:** [Your Streamlit Cloud URL Here]  
**GitHub:** [Your Repository URL Here]
