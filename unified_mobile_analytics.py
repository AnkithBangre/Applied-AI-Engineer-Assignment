import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class UnifiedMobileAnalytics:
    """
    Comprehensive mobile app analytics pipeline combining Google Play Store data
    with Apple App Store API data
    """
    
    def __init__(self, rapidapi_key: str = None):
        self.rapidapi_key = rapidapi_key
        self.playstore_data = None
        self.appstore_data = None
        self.unified_data = None
        
        # RapidAPI headers
        if rapidapi_key:
            self.headers = {
                "X-RapidAPI-Key": rapidapi_key,
                "X-RapidAPI-Host": "app-store-scraper1.p.rapidapi.com"
            }
    
    def load_playstore_data(self, file_path: str) -> bool:
        """Load and clean Google Play Store data"""
        try:
            # Load data
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} apps from Google Play Store dataset")
            
            # Clean data
            df_clean = df.copy()
            
            # Remove duplicates
            initial_count = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            print(f"Removed {initial_count - len(df_clean)} duplicates")
            
            # Clean and convert columns
            df_clean = self._clean_playstore_columns(df_clean)
            
            # Filter valid ratings
            if 'Rating' in df_clean.columns:
                df_clean = df_clean[(df_clean['Rating'] >= 1) & (df_clean['Rating'] <= 5)]
            
            self.playstore_data = df_clean
            print(f"Final cleaned dataset: {len(self.playstore_data)} apps")
            return True
            
        except Exception as e:
            print(f"Error loading Play Store data: {e}")
            return False
    
    def _clean_playstore_columns(self, df):
        """Clean specific Play Store columns"""
        df_clean = df.copy()
        
        # Clean Size column
        if 'Size' in df_clean.columns:
            df_clean['Size_MB'] = df_clean['Size'].apply(self._convert_size_to_mb)
        
        # Clean Installs column
        if 'Installs' in df_clean.columns:
            df_clean['Installs_Numeric'] = df_clean['Installs'].apply(self._convert_installs)
        
        # Clean Price column
        if 'Price' in df_clean.columns:
            df_clean['Price_USD'] = df_clean['Price'].apply(self._convert_price)
        
        # Clean Reviews column
        if 'Reviews' in df_clean.columns:
            df_clean['Reviews'] = pd.to_numeric(df_clean['Reviews'], errors='coerce')
        
        # Parse Last Updated
        if 'Last Updated' in df_clean.columns:
            df_clean['Last_Updated_Date'] = pd.to_datetime(df_clean['Last Updated'], errors='coerce')
        
        return df_clean
    
    def _convert_size_to_mb(self, size_str):
        """Convert size string to MB"""
        if pd.isna(size_str) or size_str == 'Varies with device':
            return np.nan
        
        size_str = str(size_str).upper().strip()
        
        # Extract numeric value
        import re
        numbers = re.findall(r'[\d.]+', size_str)
        if not numbers:
            return np.nan
        
        value = float(numbers[0])
        
        if 'K' in size_str:
            return value / 1024  # KB to MB
        elif 'G' in size_str:
            return value * 1024  # GB to MB
        elif 'M' in size_str:
            return value  # Already in MB
        else:
            return value / (1024 * 1024)  # Bytes to MB
    
    def _convert_installs(self, install_str):
        """Convert install string to numeric"""
        if pd.isna(install_str):
            return np.nan
        
        install_str = str(install_str).replace(',', '').replace('+', '').replace('Free', '0')
        
        try:
            return int(install_str)
        except:
            return np.nan
    
    def _convert_price(self, price_str):
        """Convert price string to USD numeric"""
        if pd.isna(price_str) or str(price_str) == '0':
            return 0.0
        
        price_str = str(price_str).replace('$', '').replace('€', '').replace('₹', '')
        
        try:
            return float(price_str)
        except:
            return 0.0
    
    def fetch_appstore_data(self, categories: List[str] = None, limit: int = 100) -> bool:
        """Fetch App Store data using RapidAPI"""
        if not self.rapidapi_key:
            print("RapidAPI key required for App Store data")
            return False
        
        if categories is None:
            categories = ['games', 'business', 'productivity', 'social', 'entertainment']
        
        appstore_apps = []
        
        for category in categories:
            print(f"Fetching {category} apps from App Store...")
            
            # Search apps in category
            apps_data = self._search_appstore_category(category, limit // len(categories))
            
            if apps_data:
                appstore_apps.extend(apps_data)
            
            # Rate limiting
            time.sleep(2)
        
        if appstore_apps:
            self.appstore_data = pd.DataFrame(appstore_apps)
            print(f"Fetched {len(self.appstore_data)} apps from App Store")
            return True
        
        return False
    
    def _search_appstore_category(self, category: str, limit: int):
        """Search App Store by category"""
        url = "https://app-store-scraper1.p.rapidapi.com/search"
        
        querystring = {
            "term": category,
            "country": "us",
            "limit": str(limit)
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=querystring)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            else:
                print(f"API Error for {category}: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Request failed for {category}: {e}")
            return []
    
    def create_unified_dataset(self):
        """Create unified dataset from both platforms"""
        unified_apps = []
        
        # Process Google Play Store data
        if self.playstore_data is not None:
            for _, app in self.playstore_data.iterrows():
                unified_app = {
                    'app_name': app.get('App', ''),
                    'platform': 'Google Play',
                    'category': app.get('Category', ''),
                    'rating': app.get('Rating', None),
                    'review_count': app.get('Reviews', None),
                    'price_usd': app.get('Price_USD', 0),
                    'size_mb': app.get('Size_MB', None),
                    'installs': app.get('Installs_Numeric', None),
                    'last_updated': app.get('Last_Updated_Date', None),
                    'content_rating': app.get('Content Rating', ''),
                    'developer': app.get('Developer', ''),
                    'type': app.get('Type', 'Free')
                }
                unified_apps.append(unified_app)
        
        # Process App Store data
        if self.appstore_data is not None:
            for _, app in self.appstore_data.iterrows():
                unified_app = {
                    'app_name': app.get('trackName', ''),
                    'platform': 'App Store',
                    'category': app.get('primaryGenreName', ''),
                    'rating': app.get('averageUserRating', None),
                    'review_count': app.get('userRatingCount', None),
                    'price_usd': app.get('price', 0),
                    'size_mb': app.get('fileSizeBytes', 0) / (1024 * 1024) if app.get('fileSizeBytes') else None,
                    'installs': None,  # Not available in App Store
                    'last_updated': pd.to_datetime(app.get('currentVersionReleaseDate', None)),
                    'content_rating': app.get('contentAdvisoryRating', ''),
                    'developer': app.get('artistName', ''),
                    'type': 'Paid' if app.get('price', 0) > 0 else 'Free'
                }
                unified_apps.append(unified_app)
        
        self.unified_data = pd.DataFrame(unified_apps)
        print(f"Created unified dataset with {len(self.unified_data)} apps")
        
        return self.unified_data
    
    def analyze_market_comparison(self):
        """Compare markets between platforms"""
        if self.unified_data is None:
            print("Please create unified dataset first!")
            return
        
        print("=== CROSS-PLATFORM MARKET ANALYSIS ===")
        
        # Platform distribution
        platform_dist = self.unified_data['platform'].value_counts()
        print("Platform Distribution:")
        print(platform_dist)
        print()
        
        # Category comparison
        print("Top Categories by Platform:")
        for platform in self.unified_data['platform'].unique():
            platform_data = self.unified_data[self.unified_data['platform'] == platform]
            top_categories = platform_data['category'].value_counts().head(5)
            print(f"\n{platform}:")
            print(top_categories)
        
        # Rating comparison
        if 'rating' in self.unified_data.columns:
            rating_by_platform = self.unified_data.groupby('platform')['rating'].agg(['mean', 'count']).round(3)
            print(f"\nRating Comparison:")
            print(rating_by_platform)
        
        # Price comparison
        price_by_platform = self.unified_data.groupby(['platform', 'type'])['price_usd'].agg(['mean', 'count']).round(2)
        print(f"\nPrice Comparison:")
        print(price_by_platform)
        
        return {
            'platform_distribution': platform_dist,
            'rating_comparison': rating_by_platform if 'rating' in self.unified_data.columns else None,
            'price_comparison': price_by_platform
        }
    
    def analyze_category_performance(self):
        """Analyze performance metrics by category"""
        if self.unified_data is None:
            print("Please create unified dataset first!")
            return
        
        print("=== CATEGORY PERFORMANCE ANALYSIS ===")
        
        # Filter out categories with too few apps
        category_counts = self.unified_data['category'].value_counts()
        significant_categories = category_counts[category_counts >= 5].index
        
        analysis_data = self.unified_data[self.unified_data['category'].isin(significant_categories)]
        
        # Performance metrics by category
        category_metrics = analysis_data.groupby('category').agg({
            'rating': ['mean', 'count'],
            'review_count': ['mean', 'median'],
            'price_usd': ['mean', 'max'],
            'size_mb': 'mean'
        }).round(2)
        
        category_metrics.columns = ['avg_rating', 'app_count', 'avg_reviews', 'median_reviews', 
                                  'avg_price', 'max_price', 'avg_size_mb']
        
        # Sort by average rating
        category_metrics = category_metrics.sort_values('avg_rating', ascending=False)
        
        print("Top 10 Categories by Average Rating:")
        print(category_metrics.head(10))
        
        # Engagement analysis (review count as proxy)
        print(f"\nTop 10 Categories by User Engagement (Avg Reviews):")
        engagement_ranking = category_metrics.sort_values('avg_reviews', ascending=False).head(10)
        print(engagement_ranking[['avg_rating', 'avg_reviews', 'app_count']])
        
        return category_metrics
    
    def identify_market_opportunities(self):
        """Identify potential market opportunities"""
        if self.unified_data is None:
            print("Please create unified dataset first!")
            return
        
        print("=== MARKET OPPORTUNITY ANALYSIS ===")
        
        # Calculate opportunity score based on various factors
        category_analysis = self.unified_data.groupby('category').agg({
            'rating': ['mean', 'count'],
            'review_count': 'mean',
            'price_usd': 'mean'
        }).round(2)
        
        category_analysis.columns = ['avg_rating', 'app_count', 'avg_reviews', 'avg_price']
        
        # Opportunity scoring
        # High opportunity = Low competition + Decent engagement + Room for quality improvement
        category_analysis['competition_score'] = 1 / (category_analysis['app_count'] / 100)  # Lower is better
        category_analysis['engagement_score'] = category_analysis['avg_reviews'] / 10000  # Higher is better
        category_analysis['quality_gap_score'] = (5 - category_analysis['avg_rating']) * 2  # Higher gap = more opportunity
        
        # Normalize scores
        for col in ['competition_score', 'engagement_score', 'quality_gap_score']:
            category_analysis[col] = (category_analysis[col] - category_analysis[col].min()) / (
                category_analysis[col].max() - category_analysis[col].min())
        
        # Calculate final opportunity score
        category_analysis['opportunity_score'] = (
            category_analysis['competition_score'] * 0.4 +
            category_analysis['engagement_score'] * 0.3 +
            category_analysis['quality_gap_score'] * 0.3
        )
        
        # Filter categories with reasonable app count (5-100 apps)
        opportunities = category_analysis[
            (category_analysis['app_count'] >= 5) & 
            (category_analysis['app_count'] <= 100)
        ].sort_values('opportunity_score', ascending=False)
        
        print("Top 10 Market Opportunities:")
        print(opportunities.head(10)[['avg_rating', 'app_count', 'avg_reviews', 'opportunity_score']])
        
        return opportunities
    
    def analyze_pricing_strategies(self):
        """Analyze pricing strategies across platforms and categories"""
        if self.unified_data is None:
            print("Please create unified dataset first!")
            return
        
        print("=== PRICING STRATEGY ANALYSIS ===")
        
        # Free vs Paid distribution
        pricing_dist = self.unified_data['type'].value_counts()
        print("App Type Distribution:")
        print(pricing_dist)
        print(f"Free apps: {pricing_dist.get('Free', 0)/len(self.unified_data)*100:.1f}%")
        
        # Paid app analysis
        paid_apps = self.unified_data[self.unified_data['price_usd'] > 0]
        
        if len(paid_apps) > 0:
            print(f"\nPaid App Pricing Statistics:")
            print(paid_apps['price_usd'].describe())
            
            # Price ranges
            price_bins = [0, 1, 3, 5, 10, 20, float('inf')]
            price_labels = ['$0-1', '$1-3', '$3-5', '$5-10', '$10-20', '$20+']
            paid_apps['price_range'] = pd.cut(paid_apps['price_usd'], bins=price_bins, labels=price_labels)
            
            print(f"\nPrice Range Distribution:")
            print(paid_apps['price_range'].value_counts())
            
            # Pricing by category
            category_pricing = paid_apps.groupby('category')['price_usd'].agg(['mean', 'count']).round(2)
            category_pricing = category_pricing[category_pricing['count'] >= 3]
            print(f"\nAverage Price by Category (3+ apps):")
            print(category_pricing.sort_values('mean', ascending=False).head(10))
        
        return paid_apps if len(paid_apps) > 0 else None
    
    def generate_competitive_insights(self, top_n: int = 20):
        """Generate insights about top performing apps"""
        if self.unified_data is None:
            print("Please create unified dataset first!")
            return
        
        print("=== COMPETITIVE INSIGHTS ===")
        
        # Top rated apps
        top_rated = self.unified_data.nlargest(top_n, 'rating')
        print(f"Top {top_n} Highest Rated Apps:")
        print(top_rated[['app_name', 'platform', 'category', 'rating', 'review_count']].head(10))
        
        # Most reviewed apps (engagement proxy)
        if 'review_count' in self.unified_data.columns:
            most_reviewed = self.unified_data.nlargest(top_n, 'review_count')
            print(f"\nTop {top_n} Most Reviewed Apps:")
            print(most_reviewed[['app_name', 'platform', 'category', 'rating', 'review_count']].head(10))
        
        # Success patterns analysis
        high_performers = self.unified_data[
            (self.unified_data['rating'] >= 4.0) & 
            (self.unified_data['review_count'] >= self.unified_data['review_count'].quantile(0.8))
        ]
        
        if len(high_performers) > 0:
            print(f"\nSuccess Patterns from High Performers ({len(high_performers)} apps):")
            
            # Common categories among high performers
            success_categories = high_performers['category'].value_counts().head(5)
            print("Top categories for high performers:")
            print(success_categories)
            
            # Average characteristics
            print(f"\nAverage characteristics of high performers:")
            print(f"Average rating: {high_performers['rating'].mean():.2f}")
            print(f"Average reviews: {high_performers['review_count'].mean():,.0f}")
            print(f"Average price: ${high_performers['price_usd'].mean():.2f}")
            if 'size_mb' in high_performers.columns:
                print(f"Average size: {high_performers['size_mb'].mean():.1f} MB")
        
        return high_performers
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive visualizations"""
        if self.unified_data is None:
            print("Please create unified dataset first!")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid of subplots
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Platform distribution
        ax1 = fig.add_subplot(gs[0, 0])
        platform_counts = self.unified_data['platform'].value_counts()
        ax1.pie(platform_counts.values, labels=platform_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('App Distribution by Platform', fontsize=12, fontweight='bold')
        
        # 2. Top categories
        ax2 = fig.add_subplot(gs[0, 1])
        top_categories = self.unified_data['category'].value_counts().head(10)
        ax2.barh(range(len(top_categories)), top_categories.values)
        ax2.set_yticks(range(len(top_categories)))
        ax2.set_yticklabels(top_categories.index, fontsize=8)
        ax2.set_title('Top 10 Categories', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Apps')
        
        # 3. Rating distribution
        ax3 = fig.add_subplot(gs[0, 2])
        valid_ratings = self.unified_data.dropna(subset=['rating'])
        if len(valid_ratings) > 0:
            ax3.hist(valid_ratings['rating'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_title('Rating Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Rating')
            ax3.set_ylabel('Frequency')
        
        # 4. Price distribution (free vs paid)
        ax4 = fig.add_subplot(gs[1, 0])
        type_counts = self.unified_data['type'].value_counts()
        ax4.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Free vs Paid Apps', fontsize=12, fontweight='bold')
        
        # 5. Platform vs Rating comparison
        ax5 = fig.add_subplot(gs[1, 1])
        platform_ratings = []
        platforms = self.unified_data['platform'].unique()
        for platform in platforms:
            platform_data = self.unified_data[self.unified_data['platform'] == platform]
            platform_ratings.append(platform_data['rating'].dropna())
        
        if platform_ratings:
            ax5.boxplot(platform_ratings, labels=platforms)
            ax5.set_title('Rating Distribution by Platform', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Rating')
        
        # 6. Size distribution
        ax6 = fig.add_subplot(gs[1, 2])
        size_data = self.unified_data.dropna(subset=['size_mb'])
        if len(size_data) > 0 and size_data['size_mb'].max() > 0:
            # Filter out extreme outliers for better visualization
            size_filtered = size_data[size_data['size_mb'] <= size_data['size_mb'].quantile(0.95)]
            ax6.hist(size_filtered['size_mb'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            ax6.set_title('App Size Distribution', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Size (MB)')
            ax6.set_ylabel('Frequency')
        
        # 7. Category vs Rating heatmap
        ax7 = fig.add_subplot(gs[2, :2])
        category_platform = pd.crosstab(self.unified_data['category'], self.unified_data['platform'], values=self.unified_data['rating'], aggfunc='mean')
        if not category_platform.empty:
            sns.heatmap(category_platform.head(15), annot=True, fmt='.2f', cmap='YlOrRd', ax=ax7)
            ax7.set_title('Average Rating by Category and Platform', fontsize=12, fontweight='bold')
        
        # 8. Review count vs Rating scatter
        ax8 = fig.add_subplot(gs[2, 2])
        scatter_data = self.unified_data.dropna(subset=['rating', 'review_count'])
        if len(scatter_data) > 0:
            # Sample data for better performance if dataset is large
            if len(scatter_data) > 1000:
                scatter_data = scatter_data.sample(1000)
            
            colors = ['blue' if p == 'Google Play' else 'red' for p in scatter_data['platform']]
            ax8.scatter(scatter_data['review_count'], scatter_data['rating'], alpha=0.6, c=colors, s=20)
            ax8.set_xlabel('Review Count (log scale)')
            ax8.set_ylabel('Rating')
            ax8.set_xscale('log')
            ax8.set_title('Rating vs Review Count', fontsize=12, fontweight='bold')
            ax8.legend(['Google Play', 'App Store'])
        
        # 9. Price analysis
        ax9 = fig.add_subplot(gs[3, :])
        paid_apps = self.unified_data[self.unified_data['price_usd'] > 0]
        if len(paid_apps) > 0:
            # Price distribution by category (top categories only)
            top_paid_categories = paid_apps['category'].value_counts().head(8).index
            paid_category_data = paid_apps[paid_apps['category'].isin(top_paid_categories)]
            
            if len(paid_category_data) > 0:
                paid_category_data.boxplot(column='price_usd', by='category', ax=ax9)
                ax9.set_title('Price Distribution by Category (Top 8 Categories)', fontsize=12, fontweight='bold')
                ax9.set_xlabel('Category')
                ax9.set_ylabel('Price (USD)')
                plt.setp(ax9.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Comprehensive Mobile App Market Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def export_insights_report(self, filename: str = 'mobile_app_insights_report.json'):
        """Export comprehensive insights as JSON report"""
        if self.unified_data is None:
            print("Please create unified dataset first!")
            return
        
        # Generate all insights
        market_comparison = self.analyze_market_comparison()
        category_performance = self.analyze_category_performance()
        opportunities = self.identify_market_opportunities()
        competitive_insights = self.generate_competitive_insights()
        
        # Compile report
        report = {
            'report_generated': datetime.now().isoformat(),
            'dataset_summary': {
                'total_apps': len(self.unified_data),
                'google_play_apps': len(self.unified_data[self.unified_data['platform'] == 'Google Play']),
                'app_store_apps': len(self.unified_data[self.unified_data['platform'] == 'App Store']),
                'unique_categories': self.unified_data['category'].nunique(),
                'unique_developers': self.unified_data['developer'].nunique()
            },
            'market_insights': {
                'platform_distribution': market_comparison['platform_distribution'].to_dict(),
                'top_categories': self.unified_data['category'].value_counts().head(10).to_dict(),
                'avg_rating_overall': float(self.unified_data['rating'].mean()),
                'free_vs_paid_ratio': self.unified_data['type'].value_counts().to_dict()
            },
            'top_opportunities': opportunities.head(10).to_dict('records') if opportunities is not None else [],
            'success_patterns': {
                'high_performer_categories': competitive_insights['category'].value_counts().head(5).to_dict(),
                'avg_rating_high_performers': float(competitive_insights['rating'].mean()),
                'avg_reviews_high_performers': float(competitive_insights['review_count'].mean())
            },
            'recommendations': self._generate_recommendations()
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Comprehensive insights report exported to: {filename}")
        return report
    
    def _generate_recommendations(self):
        """Generate strategic recommendations based on analysis"""
        recommendations = []
        
        if self.unified_data is not None:
            # Category recommendations
            category_counts = self.unified_data['category'].value_counts()
            underserved_categories = category_counts[category_counts <= 10].head(5)
            
            for category in underserved_categories.index:
                recommendations.append({
                    'type': 'Market Opportunity',
                    'category': category,
                    'recommendation': f'Consider entering {category} market - low competition with only {category_counts[category]} apps',
                    'priority': 'High' if category_counts[category] <= 5 else 'Medium'
                })
            
            # Pricing recommendations
            avg_paid_price = self.unified_data[self.unified_data['price_usd'] > 0]['price_usd'].mean()
            recommendations.append({
                'type': 'Pricing Strategy',
                'recommendation': f'Average paid app price is ${avg_paid_price:.2f}. Consider competitive pricing in this range.',
                'priority': 'Medium'
            })
            
            # Quality recommendations
            avg_rating = self.unified_data['rating'].mean()
            if avg_rating < 4.0:
                recommendations.append({
                    'type': 'Quality Focus',
                    'recommendation': f'Market average rating is {avg_rating:.2f}. Focus on quality to differentiate (target 4.0+)',
                    'priority': 'High'
                })
        
        return recommendations

# Usage Example and Testing
def run_comprehensive_analysis(csv_file_path: str, rapidapi_key: str = None):
    """
    Complete analysis workflow
    """
    print("🚀 Starting Comprehensive Mobile App Analysis...")
    
    # Initialize analyzer
    analyzer = UnifiedMobileAnalytics(rapidapi_key)
    
    # Step 1: Load Play Store data
    print("\n📱 Loading Google Play Store data...")
    if not analyzer.load_playstore_data(csv_file_path):
        print("❌ Failed to load Play Store data")
        return
    
    # Step 2: Fetch App Store data (if API key provided)
    if rapidapi_key:
        print("\n🍎 Fetching App Store data...")
        analyzer.fetch_appstore_data(['games', 'productivity', 'business', 'social', 'entertainment'])
    
    # Step 3: Create unified dataset
    print("\n🔄 Creating unified dataset...")
    analyzer.create_unified_dataset()
    
    # Step 4: Run comprehensive analysis
    print("\n📊 Running market analysis...")
    analyzer.analyze_market_comparison()
    
    print("\n🎯 Analyzing category performance...")
    analyzer.analyze_category_performance()
    
    print("\n💡 Identifying market opportunities...")
    analyzer.identify_market_opportunities()
    
    print("\n💰 Analyzing pricing strategies...")
    analyzer.analyze_pricing_strategies()
    
    print("\n🏆 Generating competitive insights...")
    analyzer.generate_competitive_insights()
    
    # Step 5: Create visualizations
    print("\n📈 Creating dashboard...")
    analyzer.create_comprehensive_dashboard()
    
    # Step 6: Export report
    print("\n📄 Exporting insights report...")
    report = analyzer.export_insights_report()
    
    print("\n✅ Analysis complete! Check the generated report and visualizations.")
    return analyzer, report

# Example usage:
"""
# Run with Google Play Store data only
analyzer, report = run_comprehensive_analysis('googleplaystore.csv')

# Run with both platforms (requires RapidAPI key)
analyzer, report = run_comprehensive_analysis('googleplaystore.csv', 'YOUR_RAPIDAPI_KEY')
"""

print("📱 Unified Mobile Analytics Pipeline Ready!")
print("Use run_comprehensive_analysis() function to start the complete analysis workflow.")
        