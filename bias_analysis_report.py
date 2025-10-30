"""
Bias Analysis and Mitigation for NER & Sentiment Analysis Model

This script identifies potential biases in the sentiment analysis model
and demonstrates how to mitigate them using rule-based systems and fairness techniques.
"""

import json
import spacy
from collections import defaultdict, Counter
import pandas as pd

print("="*80)
print("BIAS ANALYSIS & MITIGATION REPORT")
print("NER & Sentiment Analysis Model")
print("="*80)

# Load previous results
with open('ner_sentiment_output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("\n" + "="*80)
print("PART 1: IDENTIFIED BIASES IN CURRENT MODEL")
print("="*80)

# ============================================================================
# BIAS 1: LEXICON BIAS
# ============================================================================
print("\n📊 BIAS #1: LEXICON BIAS")
print("-" * 80)

POSITIVE_WORDS = {
    'love', 'amazing', 'excellent', 'fantastic', 'brilliant', 'superb',
    'happy', 'comfortable', 'worth', 'best', 'great', 'wonderful', 'good'
}

NEGATIVE_WORDS = {
    'terrible', 'disappointing', 'cheap', 'overpriced', 'disappointed',
    'bugs', 'crashes', 'bad', 'worst', 'hate', 'awful', 'poor'
}

print("\n⚠️  Issue: Limited vocabulary coverage")
print(f"   - Positive lexicon: {len(POSITIVE_WORDS)} words")
print(f"   - Negative lexicon: {len(NEGATIVE_WORDS)} words")
print(f"   - Missing: sarcasm, context-dependent words, domain-specific terms")

print("\n⚠️  Impact:")
print("   - Reviews with uncommon sentiment words may be misclassified")
print("   - Sarcastic reviews (e.g., 'Yeah, great, another broken phone') marked positive")
print("   - Nuanced opinions reduced to binary classification")

# ============================================================================
# BIAS 2: BRAND BIAS
# ============================================================================
print("\n📊 BIAS #2: BRAND BIAS")
print("-" * 80)

brand_sentiments = defaultdict(lambda: {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0})
for result in data['detailed_results']:
    for brand in result['brands']:
        brand_sentiments[brand][result['sentiment']] += 1

print("\n⚠️  Issue: Unequal brand representation")
print("\nBrand Sentiment Distribution:")
for brand, sentiments in sorted(brand_sentiments.items(), 
                                key=lambda x: sum(x[1].values()), 
                                reverse=True)[:8]:
    total = sum(sentiments.values())
    pos_pct = (sentiments['POSITIVE'] / total * 100) if total > 0 else 0
    print(f"   {brand:30s} | Positive: {pos_pct:5.1f}% | Total: {total}")

print("\n⚠️  Impact:")
print("   - Apple: 100% positive (2/2) - potential positive bias")
print("   - Samsung/Dell/Microsoft: 100% negative - potential negative bias")
print("   - Small sample sizes lead to unreliable brand-level insights")
print("   - Could perpetuate stereotypes about brand quality")

# ============================================================================
# BIAS 3: ENTITY EXTRACTION BIAS
# ============================================================================
print("\n📊 BIAS #3: ENTITY EXTRACTION BIAS")
print("-" * 80)

KNOWN_BRANDS = {
    'Apple', 'Samsung', 'Nike', 'Adidas', 'Sony', 'Dell', 'Microsoft', 
    'Canon', 'Google', 'iPhone', 'MacBook', 'Surface', 'Pixel', 'Galaxy'
}

print("\n⚠️  Issue: Predefined brand list favors well-known companies")
print(f"   - Known brands list: {len(KNOWN_BRANDS)} brands")
print(f"   - Bias toward: Large tech companies, Western brands")
print(f"   - Missing: Smaller brands, non-English brands, emerging companies")

print("\n⚠️  Impact:")
print("   - Lesser-known brands may not be recognized")
print("   - Geographic bias (US/Western-centric brand list)")
print("   - Startup/indie brands underrepresented")

# ============================================================================
# BIAS 4: SAMPLE BIAS
# ============================================================================
print("\n📊 BIAS #4: SAMPLE BIAS")
print("-" * 80)

print("\n⚠️  Issue: Non-representative dataset")
print(f"   - Total reviews: {len(data['detailed_results'])}")
print(f"   - Product categories: Electronics only")
print(f"   - Language: English only")
print(f"   - Demographics: Unknown (no user metadata)")

print("\n⚠️  Impact:")
print("   - Model trained only on tech products")
print("   - Cannot generalize to other domains (fashion, food, services)")
print("   - English-only limits global applicability")
print("   - No consideration of reviewer demographics")

# ============================================================================
# BIAS 5: SENTIMENT INTENSITY BIAS
# ============================================================================
print("\n📊 BIAS #5: SENTIMENT INTENSITY BIAS")
print("-" * 80)

print("\n⚠️  Issue: All sentiment words weighted equally")
print("   - 'good' = 'amazing' = 'love' (all count as +1)")
print("   - 'bad' = 'terrible' = 'hate' (all count as -1)")
print("   - No consideration of intensity or emphasis")

scores = [r['sentiment_score'] for r in data['detailed_results']]
print(f"\n   Sentiment scores range: {min(scores)} to {max(scores)}")
print(f"   Average score: {sum(scores)/len(scores):.2f}")

print("\n⚠️  Impact:")
print("   - 'This is good' vs 'This is AMAZING!' treated similarly")
print("   - Misses emotional intensity")
print("   - Oversimplifies complex opinions")

# ============================================================================
# BIAS 6: CONTEXT INSENSITIVITY
# ============================================================================
print("\n📊 BIAS #6: CONTEXT INSENSITIVITY")
print("-" * 80)

print("\n⚠️  Issue: No understanding of negation or context")
print("   Examples of potential misclassification:")
print("   - 'not good' → counted as positive (contains 'good')")
print("   - 'could be better' → neutral (no strong keywords)")
print("   - 'I wanted to love it but...' → positive (contains 'love')")

print("\n⚠️  Impact:")
print("   - Negations reverse sentiment but aren't handled")
print("   - Conditional statements misinterpreted")
print("   - Sarcasm completely missed")

print("\n" + "="*80)
print("PART 2: MITIGATION STRATEGIES")
print("="*80)

# ============================================================================
# MITIGATION 1: EXPANDED LEXICON WITH INTENSITY WEIGHTS
# ============================================================================
print("\n🛡️  MITIGATION #1: EXPANDED LEXICON WITH INTENSITY WEIGHTS")
print("-" * 80)

ENHANCED_POSITIVE_LEXICON = {
    # High intensity (weight: 2)
    'amazing': 2, 'excellent': 2, 'fantastic': 2, 'brilliant': 2, 
    'superb': 2, 'outstanding': 2, 'exceptional': 2, 'perfect': 2,
    'love': 2, 'adore': 2, 'incredible': 2, 'phenomenal': 2,
    
    # Medium intensity (weight: 1.5)
    'great': 1.5, 'wonderful': 1.5, 'impressive': 1.5, 'solid': 1.5,
    'happy': 1.5, 'satisfied': 1.5, 'pleased': 1.5,
    
    # Low intensity (weight: 1)
    'good': 1, 'nice': 1, 'fine': 1, 'okay': 1, 'decent': 1,
    'comfortable': 1, 'worth': 1, 'best': 1.5
}

ENHANCED_NEGATIVE_LEXICON = {
    # High intensity (weight: 2)
    'terrible': 2, 'awful': 2, 'horrible': 2, 'worst': 2,
    'hate': 2, 'disgusting': 2, 'pathetic': 2, 'useless': 2,
    
    # Medium intensity (weight: 1.5)
    'disappointing': 1.5, 'disappointed': 1.5, 'poor': 1.5,
    'bad': 1.5, 'inferior': 1.5, 'subpar': 1.5,
    
    # Low intensity (weight: 1)
    'cheap': 1, 'overpriced': 1, 'mediocre': 1, 'lacking': 1,
    'bugs': 1, 'crashes': 1, 'issues': 1
}

print("\n✅ Solution: Weighted sentiment lexicon")
print(f"   - Positive words: {len(ENHANCED_POSITIVE_LEXICON)} (up from {len(POSITIVE_WORDS)})")
print(f"   - Negative words: {len(ENHANCED_NEGATIVE_LEXICON)} (up from {len(NEGATIVE_WORDS)})")
print(f"   - Intensity weights: 1.0 (low), 1.5 (medium), 2.0 (high)")

def enhanced_sentiment_analysis(text):
    """Enhanced sentiment analysis with weighted lexicon"""
    text_lower = text.lower()
    
    positive_score = sum(weight for word, weight in ENHANCED_POSITIVE_LEXICON.items() 
                        if word in text_lower)
    negative_score = sum(weight for word, weight in ENHANCED_NEGATIVE_LEXICON.items() 
                        if word in text_lower)
    
    if positive_score > negative_score:
        sentiment = "POSITIVE"
        score = positive_score - negative_score
    elif negative_score > positive_score:
        sentiment = "NEGATIVE"
        score = negative_score - positive_score
    else:
        sentiment = "NEUTRAL"
        score = 0
    
    return {
        'sentiment': sentiment,
        'score': score,
        'positive_score': positive_score,
        'negative_score': negative_score
    }

# Test on sample reviews
print("\n📝 Example: Enhanced scoring")
sample = "I absolutely love this amazing product! It's fantastic and excellent!"
result = enhanced_sentiment_analysis(sample)
print(f"   Text: '{sample}'")
print(f"   Positive score: {result['positive_score']:.1f} (weighted)")
print(f"   Sentiment: {result['sentiment']} (Score: {result['score']:.1f})")

# ============================================================================
# MITIGATION 2: NEGATION HANDLING
# ============================================================================
print("\n🛡️  MITIGATION #2: NEGATION HANDLING")
print("-" * 80)

NEGATIONS = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 
             'nowhere', 'none', "n't", 'hardly', 'barely', 'scarcely'}

def negation_aware_sentiment(text):
    """Sentiment analysis that handles negations"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())
    
    positive_score = 0
    negative_score = 0
    
    for i, token in enumerate(doc):
        # Check if token is in sentiment lexicon
        if token.text in ENHANCED_POSITIVE_LEXICON:
            weight = ENHANCED_POSITIVE_LEXICON[token.text]
            # Check for negation in previous 3 tokens
            negated = any(doc[j].text in NEGATIONS for j in range(max(0, i-3), i))
            if negated:
                negative_score += weight  # Flip sentiment
            else:
                positive_score += weight
                
        elif token.text in ENHANCED_NEGATIVE_LEXICON:
            weight = ENHANCED_NEGATIVE_LEXICON[token.text]
            negated = any(doc[j].text in NEGATIONS for j in range(max(0, i-3), i))
            if negated:
                positive_score += weight  # Flip sentiment
            else:
                negative_score += weight
    
    if positive_score > negative_score:
        return "POSITIVE", positive_score - negative_score
    elif negative_score > positive_score:
        return "NEGATIVE", negative_score - positive_score
    else:
        return "NEUTRAL", 0

print("\n✅ Solution: Negation detection")
print("   - Detects negation words: not, no, never, n't, etc.")
print("   - Flips sentiment within 3-word window")
print("   - Handles: 'not good', 'never amazing', 'hardly terrible'")

print("\n📝 Examples:")
test_cases = [
    "This is good",
    "This is not good",
    "This is not bad",
    "Never been disappointed"
]

for test in test_cases:
    sentiment, score = negation_aware_sentiment(test)
    print(f"   '{test}' → {sentiment} (Score: {score:.1f})")

# ============================================================================
# MITIGATION 3: FAIRNESS INDICATORS & DEMOGRAPHIC PARITY
# ============================================================================
print("\n🛡️  MITIGATION #3: FAIRNESS INDICATORS (TensorFlow-inspired)")
print("-" * 80)

print("\n✅ Solution: Implement fairness metrics")
print("   Based on TensorFlow Fairness Indicators methodology:")

print("\n   1. DEMOGRAPHIC PARITY")
print("      - Ensure sentiment distribution is similar across brands")
print("      - Monitor: P(Positive | Brand A) ≈ P(Positive | Brand B)")

# Calculate fairness metrics
brand_fairness = {}
for brand, sentiments in brand_sentiments.items():
    total = sum(sentiments.values())
    if total > 0:
        pos_rate = sentiments['POSITIVE'] / total
        brand_fairness[brand] = {
            'positive_rate': pos_rate,
            'sample_size': total
        }

# Calculate overall positive rate
overall_pos_rate = data['sentiment_distribution']['POSITIVE'] / len(data['detailed_results'])

print(f"\n   Overall positive rate: {overall_pos_rate:.2%}")
print("\n   Brand-level positive rates:")
for brand, metrics in sorted(brand_fairness.items(), 
                             key=lambda x: x[1]['sample_size'], 
                             reverse=True)[:5]:
    deviation = abs(metrics['positive_rate'] - overall_pos_rate)
    status = "⚠️" if deviation > 0.3 else "✅"
    print(f"   {status} {brand:20s}: {metrics['positive_rate']:5.1%} "
          f"(n={metrics['sample_size']}, deviation={deviation:.2%})")

print("\n   2. EQUAL OPPORTUNITY")
print("      - True Positive Rate should be similar across groups")
print("      - Ensures fair treatment of actually positive reviews")

print("\n   3. EQUALIZED ODDS")
print("      - Both TPR and FPR should be similar across groups")
print("      - Comprehensive fairness across all predictions")

# ============================================================================
# MITIGATION 4: DIVERSE TRAINING DATA
# ============================================================================
print("\n🛡️  MITIGATION #4: DIVERSE & BALANCED TRAINING DATA")
print("-" * 80)

print("\n✅ Solution: Expand dataset diversity")
print("\n   Current dataset issues:")
print(f"   - Sample size: {len(data['detailed_results'])} reviews (too small)")
print(f"   - Categories: 1 (electronics only)")
print(f"   - Languages: 1 (English only)")
print(f"   - Brands: {len(data['unique_brands'])} (tech-focused)")

print("\n   Recommended improvements:")
print("   ✓ Increase sample size to 1000+ reviews per category")
print("   ✓ Include multiple product categories:")
print("     - Electronics, Fashion, Food, Services, Books, etc.")
print("   ✓ Multi-language support:")
print("     - English, Spanish, French, German, Chinese, etc.")
print("   ✓ Diverse brand representation:")
print("     - Large corporations, SMBs, startups")
print("     - Global brands from different regions")
print("   ✓ Balanced sentiment distribution:")
print("     - 33% positive, 33% negative, 33% neutral")

# ============================================================================
# MITIGATION 5: CONTEXT-AWARE NER
# ============================================================================
print("\n🛡️  MITIGATION #5: CONTEXT-AWARE NER (spaCy-based)")
print("-" * 80)

print("\n✅ Solution: Use spaCy's advanced NER features")
print("\n   1. ENTITY LINKING")
print("      - Link extracted entities to knowledge bases")
print("      - Disambiguate: 'Apple' (company) vs 'apple' (fruit)")

print("\n   2. CUSTOM NER TRAINING")
print("      - Train spaCy model on domain-specific data")
print("      - Recognize niche brands and products")

print("\n   3. RULE-BASED PATTERNS")
print("      - Add custom patterns for product names")
print("      - Example: [BRAND] + [MODEL_NUMBER] + [PRODUCT_TYPE]")

nlp = spacy.load("en_core_web_sm")

# Add custom entity ruler
from spacy.pipeline import EntityRuler
ruler = EntityRuler(nlp, overwrite_ents=True)

patterns = [
    {"label": "PRODUCT", "pattern": [{"TEXT": {"REGEX": "^[A-Z][a-z]+"}}, 
                                     {"TEXT": {"REGEX": "^[A-Z0-9]+"}}]},
    {"label": "BRAND", "pattern": "Apple"},
    {"label": "BRAND", "pattern": "Samsung"},
    {"label": "BRAND", "pattern": "Google"},
]

ruler.add_patterns(patterns)
nlp.add_pipe(ruler, before="ner")

print("\n   4. CONFIDENCE SCORES")
print("      - Track entity extraction confidence")
print("      - Flag low-confidence predictions for review")

sample_text = "The iPhone 14 Pro from Apple is amazing!"
doc = nlp(sample_text)
print(f"\n   Example: '{sample_text}'")
for ent in doc.ents:
    print(f"   - {ent.text:20s} | {ent.label_:10s}")

# ============================================================================
# MITIGATION 6: BIAS MONITORING DASHBOARD
# ============================================================================
print("\n🛡️  MITIGATION #6: CONTINUOUS BIAS MONITORING")
print("-" * 80)

print("\n✅ Solution: Implement monitoring system")
print("\n   Metrics to track:")
print("   ✓ Sentiment distribution by brand")
print("   ✓ Entity extraction accuracy by category")
print("   ✓ False positive/negative rates")
print("   ✓ Demographic parity across groups")
print("   ✓ Model drift over time")

print("\n   Tools to use:")
print("   - TensorFlow Fairness Indicators")
print("   - What-If Tool (WIT)")
print("   - Fairlearn (Microsoft)")
print("   - AI Fairness 360 (IBM)")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("PART 3: IMPLEMENTATION RECOMMENDATIONS")
print("="*80)

recommendations = [
    {
        'priority': 'HIGH',
        'action': 'Implement negation handling',
        'effort': 'Low',
        'impact': 'High',
        'timeline': '1 week'
    },
    {
        'priority': 'HIGH',
        'action': 'Expand sentiment lexicon with weights',
        'effort': 'Medium',
        'impact': 'High',
        'timeline': '2 weeks'
    },
    {
        'priority': 'HIGH',
        'action': 'Increase dataset size and diversity',
        'effort': 'High',
        'impact': 'Very High',
        'timeline': '1 month'
    },
    {
        'priority': 'MEDIUM',
        'action': 'Add fairness metrics monitoring',
        'effort': 'Medium',
        'impact': 'Medium',
        'timeline': '2 weeks'
    },
    {
        'priority': 'MEDIUM',
        'action': 'Train custom spaCy NER model',
        'effort': 'High',
        'impact': 'Medium',
        'timeline': '3 weeks'
    },
    {
        'priority': 'LOW',
        'action': 'Implement multi-language support',
        'effort': 'Very High',
        'impact': 'Medium',
        'timeline': '2 months'
    }
]

print("\n📋 Prioritized Action Plan:\n")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. [{rec['priority']}] {rec['action']}")
    print(f"   Effort: {rec['effort']} | Impact: {rec['impact']} | Timeline: {rec['timeline']}\n")

print("="*80)
print("✅ BIAS ANALYSIS COMPLETE")
print("="*80)

# Save bias analysis results
bias_report = {
    'identified_biases': [
        'Lexicon bias (limited vocabulary)',
        'Brand bias (unequal representation)',
        'Entity extraction bias (known brands only)',
        'Sample bias (non-representative dataset)',
        'Sentiment intensity bias (equal weighting)',
        'Context insensitivity (no negation handling)'
    ],
    'mitigation_strategies': [
        'Weighted sentiment lexicon',
        'Negation detection',
        'Fairness indicators',
        'Diverse training data',
        'Context-aware NER',
        'Continuous monitoring'
    ],
    'fairness_metrics': {
        'overall_positive_rate': overall_pos_rate,
        'brand_fairness': {k: v for k, v in brand_fairness.items()}
    },
    'recommendations': recommendations
}

with open('bias_analysis_report.json', 'w', encoding='utf-8') as f:
    json.dump(bias_report, f, indent=2, ensure_ascii=False)

print("\n💾 Detailed report saved to 'bias_analysis_report.json'")
