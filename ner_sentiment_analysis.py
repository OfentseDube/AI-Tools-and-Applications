"""
Named Entity Recognition (NER) and Sentiment Analysis
Goal: Extract product names and brands, and analyze sentiment using rule-based approach
"""

import spacy
import re
from collections import defaultdict

# Sample product review data (since Iris dataset doesn't contain product reviews)
sample_reviews = [
    "I absolutely love my new iPhone 14 Pro from Apple! The camera quality is amazing.",
    "The Samsung Galaxy S23 is a terrible phone. Battery life is disappointing.",
    "Just bought Nike Air Max sneakers and they are incredibly comfortable!",
    "My Sony WH-1000XM5 headphones have excellent noise cancellation. Very happy with this purchase.",
    "The Dell XPS 15 laptop is overpriced and the keyboard feels cheap.",
    "Adidas Ultraboost running shoes are fantastic! Best running shoes I've ever owned.",
    "Disappointed with the Microsoft Surface Pro 9. Too many bugs and crashes.",
    "The Canon EOS R6 camera from Canon is absolutely brilliant for photography!",
    "Bought a MacBook Pro from Apple and it's worth every penny. Superb performance!",
    "The Google Pixel 7 is okay, nothing special but gets the job done."
]

print("="*80)
print("NAMED ENTITY RECOGNITION (NER) & SENTIMENT ANALYSIS")
print("="*80)

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
    print("\n✓ Loaded spaCy model successfully\n")
except OSError:
    print("\n⚠ Installing spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    print("✓ Model installed and loaded\n")

# Define known brands and product patterns for better extraction
KNOWN_BRANDS = {
    'Apple', 'Samsung', 'Nike', 'Adidas', 'Sony', 'Dell', 'Microsoft', 
    'Canon', 'Google', 'iPhone', 'Galaxy', 'MacBook', 'Surface', 'Pixel'
}

# Rule-based sentiment analysis
POSITIVE_WORDS = {
    'love', 'amazing', 'excellent', 'fantastic', 'brilliant', 'superb',
    'happy', 'comfortable', 'worth', 'best', 'great', 'wonderful', 'good'
}

NEGATIVE_WORDS = {
    'terrible', 'disappointing', 'cheap', 'overpriced', 'disappointed',
    'bugs', 'crashes', 'bad', 'worst', 'hate', 'awful', 'poor'
}

def extract_entities(text, doc):
    """Extract product names and brands using NER and pattern matching"""
    entities = {
        'brands': set(),
        'products': set(),
        'organizations': set()
    }
    
    # Extract entities using spaCy NER
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'GPE']:
            # Check if it's a known brand
            if any(brand.lower() in ent.text.lower() for brand in KNOWN_BRANDS):
                entities['brands'].add(ent.text)
            elif ent.label_ == 'PRODUCT':
                entities['products'].add(ent.text)
            elif ent.label_ == 'ORG':
                entities['organizations'].add(ent.text)
    
    # Additional pattern matching for product names
    # Pattern: Brand + Model (e.g., "iPhone 14 Pro", "Galaxy S23")
    product_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+([A-Z0-9]+(?:\s+[A-Z][a-z]+)*)\b'
    matches = re.findall(product_pattern, text)
    for match in matches:
        full_product = ' '.join(match)
        if any(brand in full_product for brand in KNOWN_BRANDS):
            entities['products'].add(full_product)
    
    # Extract known brands directly
    for brand in KNOWN_BRANDS:
        if brand.lower() in text.lower():
            entities['brands'].add(brand)
    
    return entities

def analyze_sentiment(text):
    """Rule-based sentiment analysis"""
    text_lower = text.lower()
    
    # Count positive and negative words
    positive_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
    negative_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
    
    # Determine sentiment
    if positive_count > negative_count:
        sentiment = "POSITIVE"
        score = positive_count - negative_count
    elif negative_count > positive_count:
        sentiment = "NEGATIVE"
        score = negative_count - positive_count
    else:
        sentiment = "NEUTRAL"
        score = 0
    
    return {
        'sentiment': sentiment,
        'score': score,
        'positive_words': positive_count,
        'negative_words': negative_count
    }

# Process all reviews
results = []
all_brands = set()
all_products = set()

print("PROCESSING REVIEWS:")
print("-" * 80)

for i, review in enumerate(sample_reviews, 1):
    doc = nlp(review)
    
    # Extract entities
    entities = extract_entities(review, doc)
    
    # Analyze sentiment
    sentiment_result = analyze_sentiment(review)
    
    # Store results
    result = {
        'review_id': i,
        'text': review,
        'entities': entities,
        'sentiment': sentiment_result
    }
    results.append(result)
    
    # Aggregate brands and products
    all_brands.update(entities['brands'])
    all_products.update(entities['products'])
    
    # Display individual result
    print(f"\n📝 Review #{i}:")
    print(f"   Text: {review}")
    print(f"   Brands: {', '.join(entities['brands']) if entities['brands'] else 'None'}")
    print(f"   Products: {', '.join(entities['products']) if entities['products'] else 'None'}")
    print(f"   Sentiment: {sentiment_result['sentiment']} (Score: {sentiment_result['score']})")
    print(f"   Analysis: {sentiment_result['positive_words']} positive, {sentiment_result['negative_words']} negative words")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\n📊 Total Reviews Analyzed: {len(sample_reviews)}")
print(f"\n🏷️  Unique Brands Extracted: {len(all_brands)}")
print(f"   {', '.join(sorted(all_brands))}")

print(f"\n📦 Unique Products Extracted: {len(all_products)}")
if all_products:
    for product in sorted(all_products):
        print(f"   - {product}")

# Sentiment distribution
sentiment_counts = defaultdict(int)
for result in results:
    sentiment_counts[result['sentiment']['sentiment']] += 1

print(f"\n💭 Sentiment Distribution:")
for sentiment, count in sorted(sentiment_counts.items()):
    percentage = (count / len(sample_reviews)) * 100
    print(f"   {sentiment}: {count} ({percentage:.1f}%)")

# Most mentioned brands
brand_mentions = defaultdict(int)
for result in results:
    for brand in result['entities']['brands']:
        brand_mentions[brand] += 1

print(f"\n🔝 Top Mentioned Brands:")
for brand, count in sorted(brand_mentions.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"   {brand}: {count} mentions")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)

# Save results to file
import json

output_data = {
    'total_reviews': len(sample_reviews),
    'unique_brands': list(all_brands),
    'unique_products': list(all_products),
    'sentiment_distribution': dict(sentiment_counts),
    'detailed_results': [
        {
            'review_id': r['review_id'],
            'text': r['text'],
            'brands': list(r['entities']['brands']),
            'products': list(r['entities']['products']),
            'sentiment': r['sentiment']['sentiment'],
            'sentiment_score': r['sentiment']['score']
        }
        for r in results
    ]
}

with open('ner_sentiment_output.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("\n💾 Results saved to 'ner_sentiment_output.json'")
