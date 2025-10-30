"""
Iris Species Classification using Decision Tree

Goal: Build a decision tree classifier to predict iris species based on flower measurements.

Steps:
1. Load and explore the data
2. Preprocess the data (handle missing values, encode labels)
3. Split data into training and testing sets
4. Train a decision tree classifier
5. Evaluate using accuracy, precision, and recall
6. Visualize results
"""

# ============================================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================================

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             classification_report, confusion_matrix)
from sklearn import tree

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("Libraries imported successfully!\n")


# ============================================================================
# 2. LOAD AND EXPLORE THE DATA
# ============================================================================

# Load the Iris dataset from CSV file
df = pd.read_csv('archive (1)/Iris.csv')

print("=" * 60)
print("DATASET EXPLORATION")
print("=" * 60)

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display dataset shape
print(f"\nDataset shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check the distribution of species
print("\nSpecies distribution:")
print(df['Species'].value_counts())


# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Calculate percentage of missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("\nPercentage of missing values:")
print(missing_percentage)

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Handle missing values (if any)
# Strategy: For numerical columns, fill with median; for categorical, fill with mode

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove 'Id' column as it's not a feature
if 'Id' in numerical_cols:
    numerical_cols.remove('Id')

print(f"\nNumerical columns: {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")

# Fill missing values in numerical columns with median
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"Filled {col} missing values with median: {median_value}")

# Fill missing values in categorical columns with mode
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"Filled {col} missing values with mode: {mode_value}")

# Verify no missing values remain
print("\nMissing values after preprocessing:")
print(df.isnull().sum())

# Remove the 'Id' column as it's not useful for prediction
df_clean = df.drop('Id', axis=1)

print("\nDataset after removing 'Id' column:")
print(df_clean.head())

# Encode the target variable (Species) using LabelEncoder
# This converts categorical labels to numerical values (0, 1, 2)
label_encoder = LabelEncoder()
df_clean['Species_Encoded'] = label_encoder.fit_transform(df_clean['Species'])

# Display the mapping
print("\nSpecies encoding mapping:")
for i, species in enumerate(label_encoder.classes_):
    print(f"  {species} -> {i}")

# Display the encoded data
print("\nDataset with encoded species:")
print(df_clean[['Species', 'Species_Encoded']].head(10))


# ============================================================================
# 4. PREPARE FEATURES AND TARGET VARIABLES
# ============================================================================

print("\n" + "=" * 60)
print("FEATURE PREPARATION")
print("=" * 60)

# Separate features (X) and target variable (y)
# Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
# Target: Species_Encoded

X = df_clean[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df_clean['Species_Encoded']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

print("\nFirst 5 samples of features:")
print(X.head())

print("\nFirst 5 samples of target:")
print(y.head())


# ============================================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================

print("\n" + "=" * 60)
print("DATA SPLITTING")
print("=" * 60)

# Split the data: 80% training, 20% testing
# random_state=42 ensures reproducibility
# stratify=y ensures balanced class distribution in both sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Verify class distribution in training and testing sets
print("\nClass distribution in training set:")
print(y_train.value_counts().sort_index())

print("\nClass distribution in testing set:")
print(y_test.value_counts().sort_index())


# ============================================================================
# 6. TRAIN THE DECISION TREE CLASSIFIER
# ============================================================================

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# Initialize the Decision Tree Classifier
# criterion='gini': Uses Gini impurity for splitting
# max_depth=3: Limits tree depth to prevent overfitting
# random_state=42: Ensures reproducibility

dt_classifier = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42
)

# Train the model on the training data
print("\nTraining the Decision Tree Classifier...")
dt_classifier.fit(X_train, y_train)
print("Training completed!")

# Display tree parameters
print(f"\nTree depth: {dt_classifier.get_depth()}")
print(f"Number of leaves: {dt_classifier.get_n_leaves()}")


# ============================================================================
# 7. MAKE PREDICTIONS
# ============================================================================

print("\n" + "=" * 60)
print("MAKING PREDICTIONS")
print("=" * 60)

# Make predictions on the training set
y_train_pred = dt_classifier.predict(X_train)

# Make predictions on the testing set
y_test_pred = dt_classifier.predict(X_test)

print("\nPredictions completed!")
print(f"\nFirst 10 predictions on test set: {y_test_pred[:10]}")
print(f"Actual values for first 10 test samples: {y_test.values[:10]}")


# ============================================================================
# 8. EVALUATE MODEL PERFORMANCE
# ============================================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Calculate accuracy for training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# Calculate accuracy for testing set
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Check for overfitting
if train_accuracy - test_accuracy > 0.1:
    print("\n⚠️  Warning: Possible overfitting detected (training accuracy >> testing accuracy)")
else:
    print("\n✓ Model generalizes well to unseen data")

# Calculate precision, recall, and F1-score
# average='weighted' accounts for class imbalance
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')

print("\n=== Model Performance Metrics ===")
print(f"Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")

# Display detailed classification report
# This shows precision, recall, and F1-score for each class
print("\n=== Detailed Classification Report ===")
print(classification_report(
    y_test, 
    y_test_pred, 
    target_names=label_encoder.classes_
))


# ============================================================================
# 9. VISUALIZE RESULTS
# ============================================================================

print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# 1. Species Distribution
plt.figure(figsize=(8, 5))
df['Species'].value_counts().plot(kind='bar', color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('Distribution of Iris Species', fontsize=14, fontweight='bold')
plt.xlabel('Species', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('species_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: species_distribution.png")
plt.close()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix - Decision Tree Classifier', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")
plt.close()

# 3. Feature Importance
feature_importance = dt_classifier.feature_importances_
feature_names = X.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='#4ECDC4')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance in Decision Tree', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
plt.close()

# 4. Decision Tree Structure
plt.figure(figsize=(20, 10))
tree.plot_tree(
    dt_classifier,
    feature_names=feature_names,
    class_names=label_encoder.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree Structure', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree_structure.png', dpi=300, bbox_inches='tight')
print("✓ Saved: decision_tree_structure.png")
plt.close()

# 5. Performance Metrics Comparison
metrics = ['Accuracy', 'Precision', 'Recall']
scores = [test_accuracy, precision, recall]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
plt.ylim(0, 1.1)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, score in zip(bars, scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.4f}\n({score*100:.2f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: performance_metrics.png")
plt.close()


# ============================================================================
# 10. SUMMARY AND CONCLUSIONS
# ============================================================================

print("\n" + "=" * 60)
print("IRIS CLASSIFICATION - SUMMARY")
print("=" * 60)

print("\n📊 Dataset Information:")
print(f"   • Total samples: {len(df)}")
print(f"   • Number of features: {len(feature_names)}")
print(f"   • Number of classes: {len(label_encoder.classes_)}")
print(f"   • Classes: {', '.join(label_encoder.classes_)}")

print("\n🔧 Preprocessing Steps:")
print("   • Checked for missing values (None found)")
print("   • Removed 'Id' column (not a feature)")
print("   • Encoded species labels using LabelEncoder")
print("   • Split data: 80% training, 20% testing")

print("\n🌳 Model Configuration:")
print(f"   • Algorithm: Decision Tree Classifier")
print(f"   • Criterion: Gini impurity")
print(f"   • Max depth: 3")
print(f"   • Tree depth: {dt_classifier.get_depth()}")
print(f"   • Number of leaves: {dt_classifier.get_n_leaves()}")

print("\n📈 Performance Metrics:")
print(f"   • Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   • Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   • Recall:    {recall:.4f} ({recall*100:.2f}%)")

print("\n🎯 Most Important Features:")
for idx, row in importance_df.iterrows():
    print(f"   • {row['Feature']}: {row['Importance']:.4f}")

print("\n✅ Conclusion:")
if test_accuracy >= 0.95:
    print("   The model performs excellently on the Iris dataset!")
elif test_accuracy >= 0.85:
    print("   The model performs well on the Iris dataset.")
else:
    print("   The model may need further tuning for better performance.")

print("\n" + "=" * 60)
print("SCRIPT EXECUTION COMPLETED")
print("=" * 60)
