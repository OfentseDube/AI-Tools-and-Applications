"""
Visualization script for NER and Sentiment Analysis results
Creates visual charts and graphs from the analysis
"""

import json
import matplotlib.pyplot as plt
from collections import Counter

# Load results
with open('ner_sentiment_output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 10))

# 1. Sentiment Distribution Pie Chart
ax1 = plt.subplot(2, 3, 1)
sentiments = data['sentiment_distribution']
colors = ['#4CAF50', '#F44336', '#9E9E9E']
explode = (0.1, 0, 0)
ax1.pie(sentiments.values(), labels=sentiments.keys(), autopct='%1.1f%%',
        colors=colors, explode=explode, shadow=True, startangle=90)
ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')

# 2. Top Brands Bar Chart
ax2 = plt.subplot(2, 3, 2)
brand_counts = Counter()
for result in data['detailed_results']:
    for brand in result['brands']:
        brand_counts[brand] += 1

top_brands = dict(brand_counts.most_common(10))
ax2.barh(list(top_brands.keys()), list(top_brands.values()), color='#2196F3')
ax2.set_xlabel('Mentions', fontweight='bold')
ax2.set_title('Top 10 Brand Mentions', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

# 3. Sentiment Scores Distribution
ax3 = plt.subplot(2, 3, 3)
scores = [r['sentiment_score'] for r in data['detailed_results']]
sentiments_list = [r['sentiment'] for r in data['detailed_results']]
colors_map = {'POSITIVE': '#4CAF50', 'NEGATIVE': '#F44336', 'NEUTRAL': '#9E9E9E'}
colors_list = [colors_map[s] for s in sentiments_list]
ax3.bar(range(1, len(scores) + 1), scores, color=colors_list, alpha=0.7)
ax3.set_xlabel('Review ID', fontweight='bold')
ax3.set_ylabel('Sentiment Score', fontweight='bold')
ax3.set_title('Sentiment Scores by Review', fontsize=14, fontweight='bold')
ax3.set_xticks(range(1, len(scores) + 1))

# 4. Entity Type Distribution
ax4 = plt.subplot(2, 3, 4)
total_brands = len(data['unique_brands'])
total_products = len(data['unique_products'])
entity_types = ['Brands', 'Products']
entity_counts = [total_brands, total_products]
ax4.bar(entity_types, entity_counts, color=['#FF9800', '#9C27B0'], alpha=0.7)
ax4.set_ylabel('Count', fontweight='bold')
ax4.set_title('Entities Extracted', fontsize=14, fontweight='bold')
for i, v in enumerate(entity_counts):
    ax4.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# 5. Review Length vs Sentiment
ax5 = plt.subplot(2, 3, 5)
review_lengths = [len(r['text'].split()) for r in data['detailed_results']]
sentiment_numeric = []
for r in data['detailed_results']:
    if r['sentiment'] == 'POSITIVE':
        sentiment_numeric.append(1)
    elif r['sentiment'] == 'NEGATIVE':
        sentiment_numeric.append(-1)
    else:
        sentiment_numeric.append(0)

scatter_colors = [colors_map[r['sentiment']] for r in data['detailed_results']]
ax5.scatter(review_lengths, sentiment_numeric, c=scatter_colors, s=100, alpha=0.6)
ax5.set_xlabel('Review Length (words)', fontweight='bold')
ax5.set_ylabel('Sentiment', fontweight='bold')
ax5.set_yticks([-1, 0, 1])
ax5.set_yticklabels(['Negative', 'Neutral', 'Positive'])
ax5.set_title('Review Length vs Sentiment', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Sentiment by Brand (for brands with reviews)
ax6 = plt.subplot(2, 3, 6)
brand_sentiments = {}
for result in data['detailed_results']:
    for brand in result['brands']:
        if brand not in brand_sentiments:
            brand_sentiments[brand] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
        brand_sentiments[brand][result['sentiment']] += 1

# Get top 5 brands by total mentions
top_5_brands = sorted(brand_sentiments.items(), 
                      key=lambda x: sum(x[1].values()), 
                      reverse=True)[:5]

brands_list = [b[0] for b in top_5_brands]
positive_counts = [b[1]['POSITIVE'] for b in top_5_brands]
negative_counts = [b[1]['NEGATIVE'] for b in top_5_brands]
neutral_counts = [b[1]['NEUTRAL'] for b in top_5_brands]

x = range(len(brands_list))
width = 0.25

ax6.bar([i - width for i in x], positive_counts, width, label='Positive', color='#4CAF50', alpha=0.7)
ax6.bar(x, negative_counts, width, label='Negative', color='#F44336', alpha=0.7)
ax6.bar([i + width for i in x], neutral_counts, width, label='Neutral', color='#9E9E9E', alpha=0.7)

ax6.set_xlabel('Brand', fontweight='bold')
ax6.set_ylabel('Count', fontweight='bold')
ax6.set_title('Sentiment by Top 5 Brands', fontsize=14, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(brands_list, rotation=45, ha='right')
ax6.legend()

plt.tight_layout()
plt.savefig('ner_sentiment_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved as 'ner_sentiment_visualization.png'")
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("VISUALIZATION SUMMARY")
print("="*80)
print(f"\n📊 Total Reviews: {data['total_reviews']}")
print(f"🏷️  Unique Brands: {len(data['unique_brands'])}")
print(f"📦 Unique Products: {len(data['unique_products'])}")
print(f"\n💭 Sentiment Breakdown:")
for sentiment, count in data['sentiment_distribution'].items():
    percentage = (count / data['total_reviews']) * 100
    print(f"   {sentiment}: {count} ({percentage:.1f}%)")
print("\n✅ Charts generated successfully!")
print("="*80)
