# 📱💰 Mobile App & D2C Analytics Pipeline

> Comprehensive analytics solution for mobile app market intelligence and D2C eCommerce optimization with AI-powered creative generation.

## 🎯 Overview

This pipeline provides end-to-end analytics for:
- **Mobile App Market Analysis** (Google Play Store + Apple App Store)
- **D2C eCommerce Funnel Optimization** 
- **SEO Opportunity Mining**
- **AI-Powered Creative Content Generation**
- **Cross-Platform Strategic Insights**

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# Install required packages
pip install pandas numpy matplotlib seaborn requests openpyxl xlrd
```

### 📁 Project Structure

```
analytics-pipeline/
├── data/
│   ├── googleplaystore.csv          # Google Play Store dataset
│   ├── d2c_ecommerce_data.xlsx      # D2C synthetic dataset
│   └── sample_data/                 # Sample datasets for testing
├── src/
│   ├── mobile_analytics.py          # Mobile app analysis classes
│   ├── d2c_analytics.py            # D2C eCommerce analysis classes  
│   └── integrated_pipeline.py       # Full pipeline integration
├── outputs/
│   ├── reports/                     # Generated JSON reports
│   ├── dashboards/                  # Saved dashboard images
│   └── creative_content/            # AI-generated content
├── README.md
└── requirements.txt
```

## 📦 Installation

### Step 1: Clone/Download Files

Save the provided Python scripts as:
- `mobile_analytics.py` - Contains PlayStoreAnalyzer and UnifiedMobileAnalytics classes
- `d2c_analytics.py` - Contains D2CAnalytics class
- `integrated_pipeline.py` - Contains IntegratedAnalyticsPipeline class

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv analytics_env
source analytics_env/bin/activate  # On Windows: analytics_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
requests>=2.28.0
openpyxl>=3.0.0
xlrd>=2.0.0
scipy>=1.9.0
```

### Step 3: Prepare Data Files

1. **Google Play Store Data**: Download from [Kaggle](https://www.kaggle.com/datasets/lava18/google-play-store-apps)
   - Save as `data/googleplaystore.csv`

2. **D2C Synthetic Data**: Use the provided Excel file
   - Save as `data/d2c_ecommerce_data.xlsx`

3. **RapidAPI Key** (Optional): Get from [RapidAPI App Store Scraper](https://rapidapi.com/DataCrawler/api/app-store-scraper1/)

## 🔧 Usage Examples

### Option 1: Complete Integrated Analysis

```python
# main.py
from integrated_pipeline import run_full_analytics_suite

# Run complete analysis with App Store data
results = run_full_analytics_suite(
    playstore_csv='data/googleplaystore.csv',
    d2c_excel='data/d2c_ecommerce_data.xlsx',
    rapidapi_key='YOUR_RAPIDAPI_KEY'  # Optional
)

print("✅ Analysis complete! Check generated reports and dashboards.")
```

### Option 2: Mobile Apps Only

```python
from mobile_analytics import PlayStoreAnalyzer, UnifiedMobileAnalytics

# Initialize analyzer
analyzer = PlayStoreAnalyzer('data/googleplaystore.csv')

# Run analysis
analyzer.load_data()
analyzer.clean_data()
analyzer.analyze_categories()
analyzer.analyze_ratings_and_reviews()
analyzer.analyze_pricing_strategy()
analyzer.create_visualizations()

# Export results
analyzer.export_cleaned_data('outputs/cleaned_playstore_data.csv')
```

### Option 3: D2C eCommerce Only

```python
from d2c_analytics import run_d2c_analysis

# Run D2C analysis
analyzer, report = run_d2c_analysis('data/d2c_ecommerce_data.xlsx')

# Access creative content
creative_outputs = analyzer.generate_ai_creative_content()

print("Ad Headlines:", creative_outputs['ad_headlines'])
print("Meta Descriptions:", creative_outputs['meta_descriptions'])
```

## 📊 Expected Outputs

### 1. Reports (JSON)
- `mobile_insights_report.json` - Mobile app market analysis
- `d2c_insights_report.json` - D2C funnel and SEO analysis  
- `integrated_analytics_report.json` - Combined strategic insights

### 2. Visualizations
- Mobile app market dashboard (ratings, categories, pricing)
- D2C performance dashboard (funnel, ROAS, SEO opportunities)
- Cross-platform comparison charts

### 3. AI-Generated Creative Content
- **Ad Headlines**: Performance-driven headlines for campaigns
- **SEO Meta Descriptions**: Optimized descriptions for high-opportunity categories
- **Product Descriptions**: Conversion-focused copy with social proof

## 🔑 Configuration Options

### Environment Variables (Optional)

```bash
# .env file
RAPIDAPI_KEY=your_rapidapi_key_here
DEFAULT_COUNTRY=us
ANALYSIS_DEPTH=comprehensive  # or 'basic'
EXPORT_FORMAT=json           # or 'csv', 'excel'
```

### Configuration in Code

```python
# Custom configuration
config = {
    'mobile_analysis': {
        'min_apps_per_category': 5,
        'rating_threshold': 4.0,
        'top_categories_count': 15
    },
    'd2c_analysis': {
        'min_funnel_data_points': 10,
        'cac_threshold': 50,
        'roas_target': 3.0
    },
    'creative_generation': {
        'headline_count': 5,
        'meta_description_length': 160,
        'product_description_style': 'conversion_focused'
    }
}
```

## 🛠️ Troubleshooting

### Common Issues

**1. "File not found" errors**
```python
import os
print("Current directory:", os.getcwd())
print("Files in data folder:", os.listdir('data/'))
```

**2. Missing dependencies**
```bash
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn requests openpyxl xlrd
```

**3. Excel file reading issues**
```python
# Try different engines
df = pd.read_excel('data/d2c_data.xlsx', engine='openpyxl')
# or
df = pd.read_excel('data/d2c_data.xlsx', engine='xlrd')
```

**4. API rate limiting**
```python
import time
time.sleep(2)  # Add delays between API calls
```

**5. Memory issues with large datasets**
```python
# Process data in chunks
chunk_size = 1000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    pass
```

### Data Format Requirements

**Google Play Store CSV should have columns:**
- `App`, `Category`, `Rating`, `Reviews`, `Size`, `Installs`, `Type`, `Price`, `Content Rating`, `Genres`, `Last Updated`, `Current Ver`, `Android Ver`

**D2C Excel should include metrics like:**
- Campaign/Category identifiers
- Spend, Impressions, Clicks, CTR
- Installs, Signups, Conversions
- Revenue data
- Search Volume, Position data (for SEO)

## 📈 Advanced Usage

### Custom Analysis Pipeline

```python
from mobile_analytics import UnifiedMobileAnalytics
from d2c_analytics import D2CAnalytics

# Custom workflow
class CustomAnalyticsPipeline:
    def __init__(self):
        self.mobile = UnifiedMobileAnalytics()
        self.d2c = D2CAnalytics('data/custom_d2c_data.xlsx')
    
    def run_custom_analysis(self):
        # Load data
        self.mobile.load_playstore_data('data/googleplaystore.csv')
        self.d2c.load_d2c_data()
        
        # Custom analysis logic here
        mobile_opportunities = self.mobile.identify_market_opportunities()
        d2c_funnel = self.d2c.analyze_funnel_performance()
        
        # Combine insights
        return self.create_custom_insights(mobile_opportunities, d2c_funnel)
```

### Batch Processing Multiple Files

```python
import glob

# Process multiple D2C files
d2c_files = glob.glob('data/d2c_*.xlsx')

all_results = {}
for file_path in d2c_files:
    analyzer, report = run_d2c_analysis(file_path)
    all_results[file_path] = report

# Compare results across files
```

## 🔒 Security & Privacy

- **API Keys**: Never commit API keys to version control
- **Data Privacy**: Ensure compliance with data protection regulations
- **Rate Limiting**: Respects API rate limits to avoid blocking

## 📞 Support

### Common Questions

**Q: Can I use this without the RapidAPI key?**
A: Yes! The pipeline works with Google Play Store data alone. App Store integration is optional.

**Q: What if my D2C data has different column names?**
A: The system automatically maps common column variations. Check the `_map_columns()` function for supported formats.

**Q: How do I customize the AI-generated content?**
A: Modify the template strings in `_generate_ad_headlines()`, `_generate_meta_descriptions()`, and `_generate_product_descriptions()` functions.

**Q: Can I export results to different formats?**
A: Yes! Use pandas export functions:
```python
df.to_csv('output.csv')
df.to_excel('output.xlsx')
df.to_json('output.json')
```

### Performance Optimization

For large datasets:
```python
# Optimize pandas
import pandas as pd
pd.options.mode.chained_assignment = None  # Disable warnings
pd.set_option('display.max_columns', None)

# Use efficient data types
df = df.astype({'Rating': 'float32', 'Reviews': 'int32'})

# Sample large datasets
if len(df) > 10000:
    df_sample = df.sample(n=10000)
```

## 🎨 Customization Guide

### Adding Custom Metrics

```python
def calculate_custom_engagement_score(data):
    # Your custom logic
    engagement_score = (
        data['Reviews'] * 0.4 + 
        data['Rating'] * data['Installs'] * 0.6
    ) / 1000
    return engagement_score

# Add to analyzer
analyzer.unified_data['engagement_score'] = calculate_custom_engagement_score(analyzer.unified_data)
```

### Custom Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_custom_dashboard(data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Your custom plots
    sns.scatterplot(data=data, x='Rating', y='Reviews', ax=axes[0,0])
    # Add more plots...
    
    plt.tight_layout()
    plt.savefig('custom_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## 🏆 Best Practices

1. **Data Validation**: Always validate data quality before analysis
2. **Incremental Analysis**: Process data in manageable chunks
3. **Version Control**: Track changes to analysis parameters
4. **Documentation**: Document custom modifications
5. **Testing**: Test with sample data before full runs

## 📄 License

This analytics pipeline is provided for educational and commercial use. Please ensure compliance with data source terms of service.

---

**Ready to run your analysis?**

```bash
python main.py
```

🚀 **Happy Analyzing!**
