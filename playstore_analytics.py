import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PlayStoreAnalyzer:
    def __init__(self, file_path):
        """Initialize the analyzer with the dataset"""
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None
        
    def load_data(self):
        """Load the Google Play Store dataset"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
            print(f"\nColumns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def explore_data(self):
        """Initial data exploration"""
        if self.df is None:
            print("Please load data first!")
            return
        
        print("=== DATA OVERVIEW ===")
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\nData Types:")
        print(self.df.dtypes)
        
        print(f"\nMissing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])
        
        print(f"\nSample Data:")
        print(self.df.head())
        
        # Check for duplicates
        print(f"\nDuplicates: {self.df.duplicated().sum()}")
        
        return self.df.describe(include='all')
    
    def clean_data(self):
        """Clean and preprocess the dataset"""
        if self.df is None:
            print("Please load data first!")
            return
        
        print("=== CLEANING DATA ===")
        self.cleaned_df = self.df.copy()
        initial_rows = len(self.cleaned_df)
        
        # Remove duplicates
        self.cleaned_df = self.cleaned_df.drop_duplicates()
        print(f"Removed {initial_rows - len(self.cleaned_df)} duplicate rows")
        
        # Clean common columns that are typically messy
        if 'Size' in self.cleaned_df.columns:
            self.cleaned_df['Size'] = self._clean_size_column()
        
        if 'Installs' in self.cleaned_df.columns:
            self.cleaned_df['Installs'] = self._clean_installs_column()
        
        if 'Price' in self.cleaned_df.columns:
            self.cleaned_df['Price'] = self._clean_price_column()
        
        if 'Rating' in self.cleaned_df.columns:
            # Remove invalid ratings (outside 1-5 range)
            self.cleaned_df = self.cleaned_df[
                (self.cleaned_df['Rating'] >= 1) & (self.cleaned_df['Rating'] <= 5)
            ]
        
        if 'Reviews' in self.cleaned_df.columns:
            # Convert Reviews to numeric
            self.cleaned_df['Reviews'] = pd.to_numeric(self.cleaned_df['Reviews'], errors='coerce')
        
        # Handle 'Last Updated' column
        if 'Last Updated' in self.cleaned_df.columns:
            self.cleaned_df['Last Updated'] = pd.to_datetime(self.cleaned_df['Last Updated'], errors='coerce')
        
        print(f"Final dataset shape after cleaning: {self.cleaned_df.shape}")
        return self.cleaned_df
    
    def _clean_size_column(self):
        """Clean the Size column"""
        size_col = self.cleaned_df['Size'].copy()
        
        # Convert to numeric (MB)
        def convert_size(size):
            if pd.isna(size) or size == 'Varies with device':
                return np.nan
            
            size = str(size).upper()
            if 'M' in size:
                return float(re.sub(r'[^\d.]', '', size))
            elif 'K' in size:
                return float(re.sub(r'[^\d.]', '', size)) / 1024
            elif 'G' in size:
                return float(re.sub(r'[^\d.]', '', size)) * 1024
            else:
                return np.nan
        
        return size_col.apply(convert_size)
    
    def _clean_installs_column(self):
        """Clean the Installs column"""
        installs_col = self.cleaned_df['Installs'].copy()
        
        def convert_installs(install):
            if pd.isna(install):
                return np.nan
            
            install = str(install).replace(',', '').replace('+', '')
            try:
                return int(install)
            except:
                return np.nan
        
        return installs_col.apply(convert_installs)
    
    def _clean_price_column(self):
        """Clean the Price column"""
        price_col = self.cleaned_df['Price'].copy()
        
        def convert_price(price):
            if pd.isna(price) or price == '0':
                return 0.0
            
            price = str(price).replace('$', '')
            try:
                return float(price)
            except:
                return 0.0
        
        return price_col.apply(convert_price)
    
    def analyze_categories(self):
        """Analyze app categories"""
        if self.cleaned_df is None:
            print("Please clean data first!")
            return
        
        print("=== CATEGORY ANALYSIS ===")
        
        if 'Category' not in self.cleaned_df.columns:
            print("Category column not found!")
            return
        
        # Category distribution
        category_counts = self.cleaned_df['Category'].value_counts()
        print("Top 10 Categories by App Count:")
        print(category_counts.head(10))
        
        # Category performance metrics
        if 'Rating' in self.cleaned_df.columns:
            category_ratings = self.cleaned_df.groupby('Category')['Rating'].agg(['mean', 'count']).round(2)
            category_ratings = category_ratings[category_ratings['count'] >= 10]  # Filter categories with <10 apps
            print("\nTop 10 Categories by Average Rating:")
            print(category_ratings.sort_values('mean', ascending=False).head(10))
        
        return category_counts, category_ratings if 'Rating' in self.cleaned_df.columns else None
    
    def analyze_ratings_and_reviews(self):
        """Analyze ratings and reviews patterns"""
        if self.cleaned_df is None:
            print("Please clean data first!")
            return
        
        print("=== RATINGS & REVIEWS ANALYSIS ===")
        
        if 'Rating' in self.cleaned_df.columns:
            print(f"Rating Statistics:")
            print(self.cleaned_df['Rating'].describe())
            
            # Rating distribution
            rating_dist = self.cleaned_df['Rating'].value_counts().sort_index()
            print(f"\nRating Distribution:")
            print(rating_dist)
        
        if 'Reviews' in self.cleaned_df.columns:
            print(f"\nReview Count Statistics:")
            print(self.cleaned_df['Reviews'].describe())
            
            # High-engagement apps
            high_engagement = self.cleaned_df.nlargest(10, 'Reviews')[['App', 'Category', 'Rating', 'Reviews']]
            print(f"\nTop 10 Most Reviewed Apps:")
            print(high_engagement)
        
        return self.cleaned_df[['Rating', 'Reviews']].describe() if 'Rating' in self.cleaned_df.columns else None
    
    def analyze_pricing_strategy(self):
        """Analyze pricing strategies"""
        if self.cleaned_df is None:
            print("Please clean data first!")
            return
        
        print("=== PRICING ANALYSIS ===")
        
        if 'Price' not in self.cleaned_df.columns:
            print("Price column not found!")
            return
        
        # Free vs Paid distribution
        free_apps = (self.cleaned_df['Price'] == 0).sum()
        paid_apps = (self.cleaned_df['Price'] > 0).sum()
        
        print(f"Free Apps: {free_apps} ({free_apps/len(self.cleaned_df)*100:.1f}%)")
        print(f"Paid Apps: {paid_apps} ({paid_apps/len(self.cleaned_df)*100:.1f}%)")
        
        # Paid app pricing analysis
        if paid_apps > 0:
            paid_df = self.cleaned_df[self.cleaned_df['Price'] > 0]
            print(f"\nPaid App Pricing Statistics:")
            print(paid_df['Price'].describe())
            
            # Price ranges
            price_ranges = pd.cut(paid_df['Price'], bins=[0, 1, 5, 10, 20, float('inf')], 
                                labels=['$0-1', '$1-5', '$5-10', '$10-20', '$20+'])
            print(f"\nPrice Range Distribution:")
            print(price_ranges.value_counts())
        
        return {'free_count': free_apps, 'paid_count': paid_apps}
    
    def analyze_size_patterns(self):
        """Analyze app size patterns"""
        if self.cleaned_df is None or 'Size' not in self.cleaned_df.columns:
            print("Size data not available!")
            return
        
        print("=== SIZE ANALYSIS ===")
        
        size_data = self.cleaned_df.dropna(subset=['Size'])
        if len(size_data) == 0:
            print("No valid size data found!")
            return
        
        print(f"Size Statistics (MB):")
        print(size_data['Size'].describe())
        
        # Size categories
        size_categories = pd.cut(size_data['Size'], 
                               bins=[0, 10, 50, 100, 500, float('inf')],
                               labels=['Small (<10MB)', 'Medium (10-50MB)', 
                                     'Large (50-100MB)', 'Very Large (100-500MB)', 'Huge (500MB+)'])
        
        print(f"\nSize Category Distribution:")
        print(size_categories.value_counts())
        
        return size_data['Size'].describe()
    
    def generate_insights(self):
        """Generate key insights from the analysis"""
        if self.cleaned_df is None:
            print("Please clean data first!")
            return
        
        print("=== KEY INSIGHTS ===")
        
        insights = []
        
        # Market saturation insights
        if 'Category' in self.cleaned_df.columns:
            top_category = self.cleaned_df['Category'].mode()[0]
            category_count = self.cleaned_df['Category'].value_counts().iloc[0]
            insights.append(f"Most saturated category: {top_category} with {category_count} apps")
        
        # Quality insights
        if 'Rating' in self.cleaned_df.columns:
            avg_rating = self.cleaned_df['Rating'].mean()
            high_rated = (self.cleaned_df['Rating'] >= 4.0).sum()
            insights.append(f"Average rating: {avg_rating:.2f}")
            insights.append(f"High-rated apps (4.0+): {high_rated} ({high_rated/len(self.cleaned_df)*100:.1f}%)")
        
        # Pricing insights
        if 'Price' in self.cleaned_df.columns:
            free_pct = (self.cleaned_df['Price'] == 0).sum() / len(self.cleaned_df) * 100
            insights.append(f"Free apps percentage: {free_pct:.1f}%")
        
        # Engagement insights
        if 'Reviews' in self.cleaned_df.columns:
            median_reviews = self.cleaned_df['Reviews'].median()
            insights.append(f"Median review count: {median_reviews:,.0f}")
        
        for insight in insights:
            print(f"• {insight}")
        
        return insights
    
    def create_visualizations(self):
        """Create key visualizations"""
        if self.cleaned_df is None:
            print("Please clean data first!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Google Play Store Apps Analysis', fontsize=16, fontweight='bold')
        
        # 1. Category distribution (top 15)
        if 'Category' in self.cleaned_df.columns:
            top_categories = self.cleaned_df['Category'].value_counts().head(15)
            axes[0, 0].barh(range(len(top_categories)), top_categories.values)
            axes[0, 0].set_yticks(range(len(top_categories)))
            axes[0, 0].set_yticklabels(top_categories.index, fontsize=8)
            axes[0, 0].set_title('Top 15 Categories by App Count')
            axes[0, 0].set_xlabel('Number of Apps')
        
        # 2. Rating distribution
        if 'Rating' in self.cleaned_df.columns:
            self.cleaned_df['Rating'].hist(bins=20, ax=axes[0, 1], alpha=0.7, color='skyblue')
            axes[0, 1].set_title('Rating Distribution')
            axes[0, 1].set_xlabel('Rating')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Price distribution (for paid apps)
        if 'Price' in self.cleaned_df.columns:
            paid_apps = self.cleaned_df[self.cleaned_df['Price'] > 0]
            if len(paid_apps) > 0:
                paid_apps['Price'].hist(bins=30, ax=axes[1, 0], alpha=0.7, color='lightcoral')
                axes[1, 0].set_title('Price Distribution (Paid Apps Only)')
                axes[1, 0].set_xlabel('Price ($)')
                axes[1, 0].set_ylabel('Frequency')
        
        # 4. Size distribution
        if 'Size' in self.cleaned_df.columns:
            size_data = self.cleaned_df.dropna(subset=['Size'])
            if len(size_data) > 0:
                size_data['Size'].hist(bins=30, ax=axes[1, 1], alpha=0.7, color='lightgreen')
                axes[1, 1].set_title('App Size Distribution')
                axes[1, 1].set_xlabel('Size (MB)')
                axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def export_cleaned_data(self, output_path='cleaned_playstore_data.csv'):
        """Export cleaned dataset"""
        if self.cleaned_df is None:
            print("Please clean data first!")
            return
        
        self.cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned dataset exported to: {output_path}")
        return output_path

# Usage example:
"""
# Initialize analyzer
analyzer = PlayStoreAnalyzer('googleplaystore.csv')

# Load and explore data
analyzer.load_data()
analyzer.explore_data()

# Clean data
analyzer.clean_data()

# Run analysis
analyzer.analyze_categories()
analyzer.analyze_ratings_and_reviews()
analyzer.analyze_pricing_strategy()
analyzer.analyze_size_patterns()

# Generate insights
analyzer.generate_insights()

# Create visualizations
analyzer.create_visualizations()

# Export cleaned data
analyzer.export_cleaned_data()
"""

# Additional utility functions for API integration
class AppStoreAPIIntegrator:
    """Class to handle App Store API integration"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://app-store-scraper1.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "app-store-scraper1.p.rapidapi.com"
        }
    
    def search_apps(self, query, country='us', limit=50):
        """Search for apps using the API"""
        import requests
        import time
        
        url = f"{self.base_url}/search"
        params = {
            "term": query,
            "country": country,
            "limit": limit
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            time.sleep(1)  # Rate limiting
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    def get_app_details(self, app_id, country='us'):
        """Get detailed app information"""
        import requests
        import time
        
        url = f"{self.base_url}/app"
        params = {
            "id": app_id,
            "country": country
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            time.sleep(1)  # Rate limiting
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    def create_unified_schema(self, playstore_data, appstore_data):
        """Create unified schema for both datasets"""
        unified_data = []
        
        # Process Google Play Store data
        for _, app in playstore_data.iterrows():
            unified_app = {
                'app_name': app.get('App', ''),
                'platform': 'Google Play',
                'category': app.get('Category', ''),
                'rating': app.get('Rating', None),
                'review_count': app.get('Reviews', None),
                'price': app.get('Price', 0),
                'size_mb': app.get('Size', None),
                'installs': app.get('Installs', None),
                'last_updated': app.get('Last Updated', None),
                'content_rating': app.get('Content Rating', ''),
                'developer': app.get('Developer', '')
            }
            unified_data.append(unified_app)
        
        # Process App Store data (assuming similar structure)
        if appstore_data:
            for app in appstore_data:
                unified_app = {
                    'app_name': app.get('trackName', ''),
                    'platform': 'App Store',
                    'category': app.get('primaryGenreName', ''),
                    'rating': app.get('averageUserRating', None),
                    'review_count': app.get('userRatingCount', None),
                    'price': app.get('price', 0),
                    'size_mb': app.get('fileSizeBytes', 0) / (1024 * 1024) if app.get('fileSizeBytes') else None,
                    'installs': None,  # Not available in App Store API
                    'last_updated': app.get('currentVersionReleaseDate', None),
                    'content_rating': app.get('contentAdvisoryRating', ''),
                    'developer': app.get('artistName', '')
                }
                unified_data.append(unified_app)
        
        return pd.DataFrame(unified_data)

print("Google Play Store Analytics Pipeline Ready!")
print("Use PlayStoreAnalyzer class to analyze your dataset.")
print("Use AppStoreAPIIntegrator class to fetch and integrate App Store data.")