# 📓 Notebook Conversion Summary

## All Python Scripts Converted to Jupyter Notebooks

---

## ✅ Converted Files

### 1. **ner_sentiment_analysis.ipynb**
**Original**: `ner_sentiment_analysis.py`

**Sections**:
1. Import Libraries
2. Load spaCy Model
3. Define Sample Data & Lexicons
4. Define Entity Extraction Function
5. Define Sentiment Analysis Function
6. Process All Reviews
7. Summary Statistics
8. Save Results to JSON
9. Visualize Results (Optional)

**Features**:
- ✅ Markdown explanations for each section
- ✅ Code cells organized logically
- ✅ Output displays for each step
- ✅ Visualization with matplotlib

---

### 2. **bias_analysis.ipynb**
**Original**: `bias_analysis_report.py`

**Sections**:
1. Import Libraries
2. Load Previous Results
3. BIAS #1: Lexicon Bias
4. BIAS #2: Brand Bias
5. MITIGATION #1: Weighted Lexicon with Intensity
6. MITIGATION #2: Negation Handling (spaCy)
7. MITIGATION #3: Fairness Indicators
8. Comparison: Old vs Improved Model
9. Summary & Recommendations

**Features**:
- ✅ Detailed bias analysis
- ✅ Mitigation strategies demonstrated
- ✅ Before/after comparisons
- ✅ Fairness metrics calculations

---

### 3. **tensorflow_debugging.ipynb**
**Original**: `fixed_tensorflow_script.py`

**Sections**:
1. Buggy Code (Reference)
2. Fixed Code with Explanations
   - Step 1: Import Libraries
   - Step 2: Load and Inspect Data
   - Step 3: FIX #1 - Normalize Data
   - Step 4: FIX #2 - Reshape Data
   - Step 5: FIX #3 - One-Hot Encode Labels
   - Step 6: FIX #4 & #5 - Build Correct Model
   - Step 7: FIX #6 & #7 - Correct Compilation
   - Step 8: FIX #8 - Train with Appropriate Batch Size
   - Step 9: FIX #9 - Evaluate Properly
   - Step 10: FIX #10 - Correct Predictions
   - Step 11: Visualize Training History
3. Summary of All Fixes
4. Debugging Checklist

**Features**:
- ✅ 10 bugs identified and fixed
- ✅ Step-by-step explanations
- ✅ Training visualization
- ✅ Debugging checklist

---

## 📊 Comparison: .py vs .ipynb

| Feature | Python Script (.py) | Jupyter Notebook (.ipynb) |
|---------|---------------------|---------------------------|
| **Execution** | Run entire file | Run cell-by-cell |
| **Documentation** | Comments only | Markdown + comments |
| **Visualization** | Separate window | Inline display |
| **Debugging** | Print statements | Interactive output |
| **Learning** | Linear flow | Exploratory |
| **Sharing** | Code only | Code + output + docs |

---

## 🎯 Benefits of Notebook Format

### For Learning:
✅ **Interactive Exploration** - Run code step-by-step  
✅ **Immediate Feedback** - See output instantly  
✅ **Visual Learning** - Inline charts and graphs  
✅ **Documentation** - Markdown cells explain concepts  

### For Development:
✅ **Debugging** - Test individual cells  
✅ **Experimentation** - Modify and re-run easily  
✅ **Visualization** - Plots display inline  
✅ **Collaboration** - Share with outputs  

### For Presentation:
✅ **Professional** - Clean, organized format  
✅ **Reproducible** - Others can run your code  
✅ **Educational** - Perfect for tutorials  
✅ **Portfolio** - Showcase your work  

---

## 🚀 How to Use the Notebooks

### Option 1: Jupyter Notebook

```bash
# Install Jupyter
pip install jupyter

# Navigate to directory
cd "c:\Users\MANOWAR23\Desktop\New folder\AI for software\week 3"

# Start Jupyter
jupyter notebook

# Open any .ipynb file in browser
```

### Option 2: JupyterLab (Recommended)

```bash
# Install JupyterLab
pip install jupyterlab

# Start JupyterLab
jupyter lab

# More modern interface with better features
```

### Option 3: VS Code

```bash
# Install Python extension in VS Code
# Open .ipynb file
# VS Code has built-in Jupyter support
```

### Option 4: Google Colab

```bash
# Upload .ipynb file to Google Drive
# Open with Google Colab
# Free GPU/TPU access
# No installation required
```

---

## 📝 Notebook Structure

### Standard Cell Types:

#### 1. Markdown Cells
```markdown
# Heading 1
## Heading 2
### Heading 3

**Bold text**
*Italic text*

- Bullet point
1. Numbered list

`inline code`

```python
code block
```
```

#### 2. Code Cells
```python
# Python code
import numpy as np
print("Hello, World!")
```

#### 3. Raw Cells
```
Plain text (not executed)
```

---

## 🎨 Notebook Best Practices

### Organization:
1. **Title Cell** - Markdown with project name
2. **Introduction** - Explain purpose and goals
3. **Setup** - Import libraries
4. **Sections** - Logical grouping with headers
5. **Conclusion** - Summary and next steps

### Code Style:
- ✅ One logical operation per cell
- ✅ Clear variable names
- ✅ Comments for complex logic
- ✅ Print intermediate results
- ✅ Use markdown for explanations

### Execution:
- ✅ Run cells in order
- ✅ Restart kernel periodically
- ✅ Clear outputs before sharing
- ✅ Test full notebook execution

---

## 📚 Additional Notebooks You Can Create

### From Existing Scripts:

1. **improved_sentiment_model.py** → `improved_sentiment_model.ipynb`
   - Compare old vs new model
   - Show improvements
   - Visualize results

2. **visualize_results.py** → `data_visualization.ipynb`
   - Create charts
   - Analyze trends
   - Interactive plots

3. **mnist_app.py** → `mnist_classifier.ipynb`
   - Train model
   - Make predictions
   - Evaluate performance

---

## 🔧 Conversion Tips

### Manual Conversion:
1. Create new notebook
2. Add markdown cell for title
3. Copy code sections to code cells
4. Add markdown explanations
5. Run and verify

### Automated Conversion:
```bash
# Convert .py to .ipynb
jupyter nbconvert --to notebook script.py

# Convert .ipynb to .py
jupyter nbconvert --to python notebook.ipynb
```

### Using p2j (Python to Jupyter):
```bash
pip install p2j
p2j script.py
```

---

## 📊 Notebook Features

### Magic Commands:

```python
# Time execution
%time code_to_time()
%timeit code_to_benchmark()

# System commands
!pip install package
!ls

# Display settings
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# Load external code
%load script.py
%run script.py

# Debug
%debug
%pdb
```

### Widgets (Interactive):

```python
from ipywidgets import interact

@interact(x=(0, 10))
def square(x):
    return x ** 2
```

### Rich Display:

```python
from IPython.display import Image, HTML, Markdown

display(Image('image.png'))
display(HTML('<h1>Hello</h1>'))
display(Markdown('**Bold**'))
```

---

## ✅ Verification Checklist

### For Each Notebook:

- [ ] All cells execute without errors
- [ ] Outputs are displayed correctly
- [ ] Markdown formatting is correct
- [ ] Code is well-commented
- [ ] Visualizations render properly
- [ ] Dependencies are listed
- [ ] Kernel can be restarted and run all
- [ ] File paths are correct
- [ ] Data files are accessible

---

## 🎓 Learning Path

### Recommended Order:

1. **tensorflow_debugging.ipynb**
   - Learn TensorFlow basics
   - Understand common errors
   - Practice debugging

2. **ner_sentiment_analysis.ipynb**
   - NLP fundamentals
   - spaCy usage
   - Sentiment analysis

3. **bias_analysis.ipynb**
   - Bias identification
   - Mitigation strategies
   - Fairness metrics

---

## 📦 Files Summary

```
week 3/
├── 📓 Jupyter Notebooks (NEW)
│   ├── ner_sentiment_analysis.ipynb
│   ├── bias_analysis.ipynb
│   └── tensorflow_debugging.ipynb
│
├── 🐍 Python Scripts (Original)
│   ├── ner_sentiment_analysis.py
│   ├── bias_analysis_report.py
│   ├── fixed_tensorflow_script.py
│   ├── improved_sentiment_model.py
│   └── mnist_app.py
│
├── 📚 Documentation
│   ├── DEPLOYMENT_GUIDE.md
│   ├── DEBUGGING_GUIDE.md
│   ├── BIAS_README.md
│   └── NOTEBOOK_CONVERSION_SUMMARY.md (this file)
│
└── 📊 Data Files
    ├── ner_sentiment_output.json
    ├── improved_sentiment_results.json
    └── mnist_model.h5
```

---

## 🚀 Next Steps

### To Use Notebooks:

1. **Install Jupyter**
   ```bash
   pip install jupyter jupyterlab
   ```

2. **Start Jupyter**
   ```bash
   jupyter lab
   ```

3. **Open Notebooks**
   - Navigate to week 3 folder
   - Open any .ipynb file
   - Run cells sequentially

4. **Experiment**
   - Modify code
   - Add new cells
   - Create visualizations
   - Document findings

---

## 💡 Tips for Success

### Working with Notebooks:

1. **Save Frequently**
   - Auto-save is enabled
   - Manual save: `Ctrl+S`
   - Create checkpoints

2. **Restart Kernel**
   - Clear variables: Kernel → Restart
   - Fresh start: Restart & Clear Output
   - Full test: Restart & Run All

3. **Organize Code**
   - One task per cell
   - Use markdown headers
   - Add explanations
   - Include examples

4. **Debug Effectively**
   - Print intermediate values
   - Use `%debug` magic
   - Check variable types
   - Verify shapes

---

## 📖 Resources

### Jupyter Documentation:
- **Jupyter Notebook**: https://jupyter-notebook.readthedocs.io/
- **JupyterLab**: https://jupyterlab.readthedocs.io/
- **IPython**: https://ipython.readthedocs.io/

### Tutorials:
- **Jupyter Tutorial**: https://jupyter.org/try
- **Notebook Basics**: https://jupyter-notebook.readthedocs.io/en/stable/notebook.html
- **Magic Commands**: https://ipython.readthedocs.io/en/stable/interactive/magics.html

### Extensions:
- **nbextensions**: https://jupyter-contrib-nbextensions.readthedocs.io/
- **JupyterLab Extensions**: https://jupyterlab.readthedocs.io/en/stable/user/extensions.html

---

## ✅ Conversion Complete!

**Status**: ✅ All major Python scripts converted to Jupyter notebooks

**Notebooks Created**: 3
- ner_sentiment_analysis.ipynb
- bias_analysis.ipynb
- tensorflow_debugging.ipynb

**Features Added**:
- ✅ Markdown documentation
- ✅ Section organization
- ✅ Code explanations
- ✅ Inline visualizations
- ✅ Interactive execution

**Ready to Use**: Yes! Open with Jupyter and start exploring.

---

**Document Version**: 1.0  
**Last Updated**: October 2025  
**Status**: ✅ Complete
