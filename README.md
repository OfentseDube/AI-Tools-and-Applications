# Iris Species Classification using Decision Tree

This project implements a machine learning pipeline to classify iris species using a Decision Tree Classifier.

## 📋 Project Overview

**Goal:** Build a decision tree classifier to predict iris species based on flower measurements (sepal length, sepal width, petal length, and petal width).

**Dataset:** Iris dataset with 150 samples across 3 species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## 🎯 Objectives

1. **Data Preprocessing:**
   - Handle missing values (if any)
   - Encode categorical labels
   - Remove non-feature columns

2. **Model Training:**
   - Train a Decision Tree Classifier
   - Use 80/20 train-test split
   - Implement stratified sampling for balanced classes

3. **Model Evaluation:**
   - Calculate accuracy, precision, and recall
   - Generate confusion matrix
   - Analyze feature importance
   - Visualize decision tree structure

## 📁 Project Structure

```
week 3/
│
├── archive (1)/
│   └── Iris.csv                    # Dataset
│
├── iris_classification.ipynb       # Jupyter notebook (interactive)
├── iris_classification.py          # Python script (standalone)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Install required packages:**

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Project

#### Option 1: Jupyter Notebook (Recommended)

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `iris_classification.ipynb` in your browser

3. Run cells sequentially (Shift + Enter)

#### Option 2: Python Script

Run the standalone script:

```bash
python iris_classification.py
```

This will:
- Process the data
- Train the model
- Display evaluation metrics
- Save visualization plots as PNG files

## 📊 Output Files

When running the Python script, the following visualizations are generated:

- `species_distribution.png` - Distribution of iris species in the dataset
- `confusion_matrix.png` - Confusion matrix showing prediction accuracy
- `feature_importance.png` - Importance of each feature in the model
- `decision_tree_structure.png` - Visual representation of the decision tree
- `performance_metrics.png` - Bar chart of accuracy, precision, and recall

## 🔍 Key Features

### Data Preprocessing
- ✅ Missing value detection and handling
- ✅ Label encoding for species classification
- ✅ Feature selection (removing ID column)
- ✅ Stratified train-test split

### Model Configuration
- **Algorithm:** Decision Tree Classifier
- **Criterion:** Gini impurity
- **Max Depth:** 3 (to prevent overfitting)
- **Random State:** 42 (for reproducibility)

### Evaluation Metrics
- **Accuracy:** Overall correctness of predictions
- **Precision:** Proportion of correct positive predictions
- **Recall:** Proportion of actual positives correctly identified
- **Confusion Matrix:** Detailed breakdown of predictions vs. actual values

## 📈 Expected Results

The Decision Tree Classifier typically achieves:
- **Accuracy:** ~95-100%
- **Precision:** ~95-100%
- **Recall:** ~95-100%

The Iris dataset is relatively simple and well-separated, making it ideal for demonstrating classification algorithms.

## 🧪 Code Highlights

### Data Loading
```python
df = pd.read_csv('archive (1)/Iris.csv')
```

### Label Encoding
```python
label_encoder = LabelEncoder()
df_clean['Species_Encoded'] = label_encoder.fit_transform(df_clean['Species'])
```

### Model Training
```python
dt_classifier = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)
```

### Evaluation
```python
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
```

## 📚 Learning Outcomes

After completing this project, you will understand:

1. **Data Preprocessing:**
   - How to handle missing values
   - How to encode categorical variables
   - How to split data for training and testing

2. **Decision Trees:**
   - How decision trees make predictions
   - The concept of Gini impurity
   - How to prevent overfitting with max_depth

3. **Model Evaluation:**
   - Difference between accuracy, precision, and recall
   - How to interpret a confusion matrix
   - How to analyze feature importance

4. **Visualization:**
   - How to create informative plots
   - How to visualize decision tree structure
   - How to present model performance

## 🛠️ Customization

You can modify the model parameters in the code:

```python
dt_classifier = DecisionTreeClassifier(
    criterion='gini',        # Try 'entropy' for information gain
    max_depth=3,            # Increase for more complex trees
    min_samples_split=2,    # Minimum samples to split a node
    min_samples_leaf=1,     # Minimum samples in a leaf node
    random_state=42
)
```

## 📝 Notes

- The dataset is clean with no missing values
- All features are numerical (measurements in cm)
- The dataset is balanced (50 samples per species)
- The model uses stratified sampling to maintain class balance

## 🤝 Contributing

Feel free to:
- Experiment with different hyperparameters
- Try other classification algorithms (Random Forest, SVM, etc.)
- Add cross-validation for more robust evaluation
- Implement feature scaling/normalization

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created as part of AI for Software course - Week 3 assignment.

---

**Happy Learning! 🎓**
