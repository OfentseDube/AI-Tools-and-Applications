# Named Entity Recognition (NER) & Sentiment Analysis

## 📋 Project Overview

This project demonstrates **Named Entity Recognition (NER)** to extract product names and brands, combined with **rule-based sentiment analysis** to classify reviews as positive, negative, or neutral.

## 🎯 Goals Achieved

1. ✅ **Named Entity Recognition (NER)**: Extract product names and brands from text
2. ✅ **Sentiment Analysis**: Analyze sentiment using a rule-based approach
3. ✅ **Output**: Code snippet and results showing extracted entities and sentiment

## 🛠️ Technologies Used

- **spaCy**: For Named Entity Recognition (NER)
- **Python**: Core programming language
- **Regular Expressions**: For pattern matching and entity extraction
- **Rule-based Sentiment Analysis**: Custom lexicon-based approach

## 📊 Methodology

### Named Entity Recognition (NER)

The NER system uses a hybrid approach:

1. **spaCy NER Model**: Uses the `en_core_web_sm` model to identify:
   - Organizations (ORG)
   - Products (PRODUCT)
   - Geopolitical entities (GPE)

2. **Pattern Matching**: Regular expressions to capture:
   - Brand + Model patterns (e.g., "iPhone 14 Pro", "Galaxy S23")
   - Known brand names from a predefined list

3. **Brand Dictionary**: Maintains a list of known brands:
   - Apple, Samsung, Nike, Adidas, Sony, Dell, Microsoft, Canon, Google, etc.

### Sentiment Analysis (Rule-Based)

The sentiment analyzer uses lexicon-based scoring:

1. **Positive Words Lexicon**: 
   - love, amazing, excellent, fantastic, brilliant, superb, happy, comfortable, worth, best, great, wonderful, good

2. **Negative Words Lexicon**:
   - terrible, disappointing, cheap, overpriced, disappointed, bugs, crashes, bad, worst, hate, awful, poor

3. **Scoring Algorithm**:
   - Count positive and negative words in the text
   - Sentiment = POSITIVE if positive_count > negative_count
   - Sentiment = NEGATIVE if negative_count > positive_count
   - Sentiment = NEUTRAL if counts are equal
   - Score = |positive_count - negative_count|

## 📈 Results Summary

### Dataset
- **Total Reviews**: 10 product reviews
- **Data Source**: Sample product reviews (since Iris dataset contains botanical data, not product reviews)

### Extracted Entities

**Brands Identified**: 17 unique brands
- Apple, Samsung, Nike, Adidas, Sony, Dell, Microsoft, Canon, Google, iPhone, MacBook, Galaxy, Pixel, Surface, etc.

**Products Identified**: 7 unique products
- Microsoft Surface Pro 9
- Samsung Galaxy S23
- Google Pixel 7
- Canon EOS R6
- Dell XPS 15
- Sony WH-1000XM5
- Nike Air Max

### Sentiment Distribution

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| POSITIVE  | 6     | 60.0%      |
| NEGATIVE  | 3     | 30.0%      |
| NEUTRAL   | 1     | 10.0%      |

### Top Mentioned Brands

1. **Apple**: 2 mentions
2. **iPhone**: 1 mention
3. **Samsung**: 1 mention
4. **Galaxy**: 1 mention
5. **Nike**: 1 mention

## 📝 Sample Output

### Example 1: Positive Review
```
Review: "I absolutely love my new iPhone 14 Pro from Apple! The camera quality is amazing."
Brands: iPhone, Apple
Products: None
Sentiment: POSITIVE (Score: 2)
Analysis: 2 positive, 0 negative words
```

### Example 2: Negative Review
```
Review: "The Samsung Galaxy S23 is a terrible phone. Battery life is disappointing."
Brands: Samsung, Galaxy
Products: The Samsung Galaxy S23
Sentiment: NEGATIVE (Score: 2)
Analysis: 0 positive, 2 negative words
```

### Example 3: Neutral Review
```
Review: "The Google Pixel 7 is okay, nothing special but gets the job done."
Brands: Pixel, Google, The Google Pixel 7
Products: The Google Pixel 7
Sentiment: NEUTRAL (Score: 0)
Analysis: 0 positive, 0 negative words
```

## 🚀 How to Run

### Prerequisites
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Execution
```bash
python ner_sentiment_analysis.py
```

### Output Files
- **Console Output**: Detailed analysis of each review
- **ner_sentiment_output.json**: JSON file with all results

## 📂 File Structure

```
week 3/
├── ner_sentiment_analysis.py      # Main script
├── ner_sentiment_output.json      # Results in JSON format
├── NER_SENTIMENT_README.md        # This documentation
└── archive (1)/
    ├── Iris.csv                   # Original Iris dataset
    └── database.sqlite            # SQLite database with Iris data
```

## 🔍 Key Features

1. **Hybrid NER Approach**: Combines ML-based NER with rule-based pattern matching
2. **Customizable Lexicons**: Easy to extend positive/negative word lists
3. **Detailed Output**: Individual review analysis + aggregate statistics
4. **JSON Export**: Machine-readable output for further processing
5. **Visual Formatting**: Clear console output with emojis and formatting

## 📊 Insights

- **60% of reviews are positive**, indicating generally favorable product sentiment
- **Apple** is the most mentioned brand (2 mentions)
- The system successfully extracted both explicit brand names and product models
- Rule-based sentiment analysis provides interpretable results with word counts

## 🎓 Educational Value

This project demonstrates:
- Natural Language Processing (NLP) techniques
- Named Entity Recognition implementation
- Rule-based sentiment analysis
- Text preprocessing and pattern matching
- Data aggregation and statistical analysis

## 🔮 Future Enhancements

1. **Machine Learning Sentiment**: Replace rule-based with ML models (e.g., BERT, RoBERTa)
2. **Aspect-Based Sentiment**: Analyze sentiment for specific product features
3. **Real Data Integration**: Connect to actual product review APIs
4. **Visualization**: Add charts and graphs for sentiment trends
5. **Multi-language Support**: Extend to non-English reviews

## 📄 License

This is an educational project for AI/ML coursework.

---

**Author**: AI for Software - Week 3 Assignment  
**Date**: October 2025  
**Task**: Named Entity Recognition and Sentiment Analysis
