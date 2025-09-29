import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import warnings
import json
import os
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

# OpenAI integration
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Install with: pip install openai")

class PlayStoreAnalyzer:
    def __init__(self, file_path, openai_api_key=None):
        """Initialize the analyzer with the dataset and optional OpenAI integration"""
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None
        self.openai_client = None
        
        # Initialize OpenAI client
        if openai_api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                print("OpenAI integration enabled")
            except Exception as e:
                print(f"OpenAI initialization failed: {e}")
        elif not openai_api_key:
            # Try to get from environment variable
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and OPENAI_AVAILABLE:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    print("OpenAI integration enabled (from environment)")
                except Exception as e:
                    print(f"OpenAI initialization failed: {e}")
            else:
                print("OpenAI API key not provided. AI insights will be template-based.")
        
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
    
    def generate_openai_insights(self, insight_type: str = "comprehensive", max_tokens: int = 1500):
        """Generate advanced insights using OpenAI GPT models"""
        if not self.openai_client:
            print("OpenAI client not available. Using template-based insights.")
            return self.generate_insights()
        
        print("Generating OpenAI-powered insights...")
        
        # Prepare data context for OpenAI
        data_context = self._prepare_openai_context()
        
        insights = {}
        
        try:
            if insight_type in ["comprehensive", "market_analysis"]:
                insights['market_analysis'] = self._generate_market_analysis_openai(data_context, max_tokens)
            
            if insight_type in ["comprehensive", "competitive"]:
                insights['competitive_insights'] = self._generate_competitive_insights_openai(data_context, max_tokens)
            
            if insight_type in ["comprehensive", "opportunities"]:
                insights['growth_opportunities'] = self._generate_opportunities_openai(data_context, max_tokens)
            
            if insight_type in ["comprehensive", "strategy"]:
                insights['strategic_recommendations'] = self._generate_strategy_openai(data_context, max_tokens)
            
            print("OpenAI insights generated successfully!")
            return insights
            
        except Exception as e:
            print(f"Error generating OpenAI insights: {e}")
            print("Falling back to template-based insights...")
            return self.generate_insights()
    
    def _prepare_openai_context(self):
        """Prepare structured data context for OpenAI analysis"""
        if self.cleaned_df is None:
            return {}
        
        context = {
            'dataset_summary': {
                'total_apps': len(self.cleaned_df),
                'columns': list(self.cleaned_df.columns),
            }
        }
        
        # Category analysis
        if 'Category' in self.cleaned_df.columns:
            category_stats = self.cleaned_df['Category'].value_counts().head(10)
            context['top_categories'] = category_stats.to_dict()
            
            if 'Rating' in self.cleaned_df.columns:
                category_ratings = self.cleaned_df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10)
                context['top_rated_categories'] = category_ratings.to_dict()
        
        # Rating analysis
        if 'Rating' in self.cleaned_df.columns:
            context['rating_stats'] = {
                'average': float(self.cleaned_df['Rating'].mean()),
                'median': float(self.cleaned_df['Rating'].median()),
                'high_rated_percent': float((self.cleaned_df['Rating'] >= 4.0).mean() * 100)
            }
        
        # Pricing analysis
        if 'Price' in self.cleaned_df.columns:
            context['pricing_stats'] = {
                'free_apps_percent': float((self.cleaned_df['Price'] == 0).mean() * 100),
                'paid_apps_count': int((self.cleaned_df['Price'] > 0).sum()),
                'average_paid_price': float(self.cleaned_df[self.cleaned_df['Price'] > 0]['Price'].mean()) if (self.cleaned_df['Price'] > 0).any() else 0
            }
        
        # Engagement analysis
        if 'Reviews' in self.cleaned_df.columns:
            context['engagement_stats'] = {
                'median_reviews': float(self.cleaned_df['Reviews'].median()),
                'mean_reviews': float(self.cleaned_df['Reviews'].mean()),
                'top_reviewed_categories': self.cleaned_df.groupby('Category')['Reviews'].mean().sort_values(ascending=False).head(5).to_dict() if 'Category' in self.cleaned_df.columns else {}
            }
        
        # Size analysis
        if 'Size' in self.cleaned_df.columns:
            size_data = self.cleaned_df.dropna(subset=['Size'])
            if not size_data.empty:
                context['size_stats'] = {
                    'average_size_mb': float(size_data['Size'].mean()),
                    'median_size_mb': float(size_data['Size'].median()),
                    'large_apps_percent': float((size_data['Size'] > 100).mean() * 100)
                }
        
        return context
    
    def _generate_market_analysis_openai(self, context, max_tokens):
        """Generate market analysis using OpenAI"""
        prompt = f"""
        As a mobile app market analyst, analyze this Google Play Store data and provide comprehensive market insights:

        MARKET DATA:
        {json.dumps(context, indent=2)}

        Please provide:
        1. Overall market landscape analysis
        2. Category saturation and competition levels
        3. Quality trends and user expectations
        4. Pricing strategy insights
        5. User engagement patterns
        6. Market opportunities and gaps
        7. Industry trends and predictions

        Focus on actionable insights for app developers and publishers.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior mobile app market analyst with expertise in Google Play Store trends, user behavior, and competitive analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating market analysis: {e}")
            return "Market analysis generation failed. Please check your OpenAI configuration."
    
    def _generate_competitive_insights_openai(self, context, max_tokens):
        """Generate competitive insights using OpenAI"""
        prompt = f"""
        Based on this Google Play Store analysis, provide competitive intelligence insights:

        APP MARKET DATA:
        {json.dumps(context, indent=2)}

        Analyze and provide:
        1. Competitive landscape by category
        2. Success factors for top-rated apps
        3. Differentiation strategies
        4. Market positioning opportunities
        5. Competitive threats and advantages
        6. User acquisition strategies
        7. Monetization model effectiveness

        Include specific recommendations for competing in saturated vs. emerging categories.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a competitive intelligence expert specializing in mobile app markets with deep knowledge of Android ecosystem and user behavior patterns."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating competitive insights: {e}")
            return "Competitive insights generation failed. Please check your OpenAI configuration."
    
    def _generate_opportunities_openai(self, context, max_tokens):
        """Generate growth opportunities using OpenAI"""
        prompt = f"""
        Identify growth opportunities and market gaps based on this Google Play Store data:

        MARKET ANALYSIS DATA:
        {json.dumps(context, indent=2)}

        Provide insights on:
        1. Underserved market segments and categories
        2. Quality improvement opportunities
        3. Pricing optimization strategies
        4. Feature gap analysis
        5. Emerging trends and opportunities
        6. Niche market potential
        7. Innovation areas with low competition
        8. User experience improvement opportunities

        Prioritize opportunities by potential impact and feasibility.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a growth strategist and product manager specializing in mobile app development with expertise in identifying market opportunities and product gaps."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.8
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating opportunities: {e}")
            return "Opportunities analysis generation failed. Please check your OpenAI configuration."
    
    def _generate_strategy_openai(self, context, max_tokens):
        """Generate strategic recommendations using OpenAI"""
        prompt = f"""
        Develop strategic recommendations for app developers based on this Play Store analysis:

        STRATEGIC CONTEXT:
        {json.dumps(context, indent=2)}

        Provide strategic guidance on:
        1. App development priorities and focus areas
        2. Go-to-market strategies for different categories
        3. User acquisition and retention strategies
        4. Monetization optimization approaches
        5. Product roadmap recommendations
        6. Risk mitigation strategies
        7. Long-term positioning and brand building
        8. Technology and platform considerations

        Include implementation timelines and success metrics.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior product strategy consultant with extensive experience in mobile app development, monetization, and market positioning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.6
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating strategy: {e}")
            return "Strategic recommendations generation failed. Please check your OpenAI configuration."
    
    def generate_insights(self):
        """Generate key insights from the analysis (fallback method)"""
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
    
    def export_enhanced_report(self, filename: str = 'enhanced_playstore_report.json', include_openai: bool = True):
        """Export comprehensive report with OpenAI insights"""
        
        # Generate OpenAI insights if available
        openai_insights = {}
        if include_openai and self.openai_client:
            try:
                openai_insights = self.generate_openai_insights("comprehensive", max_tokens=2000)
            except Exception as e:
                print(f"Warning: Could not generate OpenAI insights: {e}")
        
        # Generate traditional insights
        basic_insights = self.generate_insights()
        
        # Compile comprehensive report
        report = {
            'report_generated': datetime.now().isoformat(),
            'openai_enabled': self.openai_client is not None,
            'dataset_summary': {
                'total_apps': len(self.cleaned_df) if self.cleaned_df is not None else 0,
                'total_categories': self.cleaned_df['Category'].nunique() if self.cleaned_df is not None and 'Category' in self.cleaned_df.columns else 0,
                'average_rating': float(self.cleaned_df['Rating'].mean()) if self.cleaned_df is not None and 'Rating' in self.cleaned_df.columns else 0,
                'free_apps_percentage': float((self.cleaned_df['Price'] == 0).mean() * 100) if self.cleaned_df is not None and 'Price' in self.cleaned_df.columns else 0
            },
            'basic_insights': basic_insights,
            'openai_insights': openai_insights,
            'category_analysis': self._get_category_summary(),
            'pricing_analysis': self._get_pricing_summary(),
            'quality_analysis': self._get_quality_summary()
        }
        
        # Export to JSON
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Enhanced Play Store report exported to: {filename}")
        return report
    
    def _get_category_summary(self):
        """Get category analysis summary"""
        if self.cleaned_df is None or 'Category' not in self.cleaned_df.columns:
            return {}
        
        return {
            'top_categories': self.cleaned_df['Category'].value_counts().head(10).to_dict(),
            'total_categories': int(self.cleaned_df['Category'].nunique()),
            'most_saturated': str(self.cleaned_df['Category'].mode()[0])
        }
    
    def _get_pricing_summary(self):
        """Get pricing analysis summary"""
        if self.cleaned_df is None or 'Price' not in self.cleaned_df.columns:
            return {}
        
        return {
            'free_apps_count': int((self.cleaned_df['Price'] == 0).sum()),
            'paid_apps_count': int((self.cleaned_df['Price'] > 0).sum()),
            'average_paid_price': float(self.cleaned_df[self.cleaned_df['Price'] > 0]['Price'].mean()) if (self.cleaned_df['Price'] > 0).any() else 0,
            'max_price': float(self.cleaned_df['Price'].max()),
            'free_percentage': float((self.cleaned_df['Price'] == 0).mean() * 100)
        }
    
    def _get_quality_summary(self):
        """Get quality analysis summary"""
        if self.cleaned_df is None or 'Rating' not in self.cleaned_df.columns:
            return {}
        
        return {
            'average_rating': float(self.cleaned_df['Rating'].mean()),
            'median_rating': float(self.cleaned_df['Rating'].median()),
            'high_rated_apps_count': int((self.cleaned_df['Rating'] >= 4.0).sum()),
            'high_rated_percentage': float((self.cleaned_df['Rating'] >= 4.0).mean() * 100),
            'low_rated_apps_count': int((self.cleaned_df['Rating'] <= 2.0).sum())
        }
    
    def export_cleaned_data(self, output_path='cleaned_playstore_data.csv'):
        """Export cleaned dataset"""
        if self.cleaned_df is None:
            print("Please clean data first!")
            return
        
        self.cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned dataset exported to: {output_path}")
        return output_path

# Enhanced main execution function
def run_enhanced_playstore_analysis(csv_file_path: str, openai_api_key: str = None):
    """
    Complete PlayStore analysis workflow with OpenAI-powered insights
    """
    print("Starting Enhanced Google Play Store Analysis with AI Insights...")
    
    # Initialize analyzer with OpenAI
    analyzer = PlayStoreAnalyzer(csv_file_path, openai_api_key)
    
    # Load and clean data
    print("\nLoading and cleaning dataset...")
    if not analyzer.load_data():
        return None
    
    analyzer.explore_data()
    analyzer.clean_data()
    
    # Run traditional analyses
    print("\nRunning market analysis...")
    analyzer.analyze_categories()
    analyzer.analyze_ratings_and_reviews()
    analyzer.analyze_pricing_strategy()
    analyzer.analyze_size_patterns()
    
    # Generate AI-powered insights
    print("\nGenerating AI-powered insights...")
    if analyzer.openai_client:
        ai_insights = analyzer.generate_openai_insights("comprehensive", max_tokens=2000)
        
        print("\n=== AI-GENERATED MARKET ANALYSIS ===")
        if 'market_analysis' in ai_insights:
            print(ai_insights['market_analysis'])
        
        print("\n=== AI-GENERATED COMPETITIVE INSIGHTS ===")
        if 'competitive_insights' in ai_insights:
            print(ai_insights['competitive_insights'])
        
        print("\n=== AI-GENERATED GROWTH OPPORTUNITIES ===")
        if 'growth_opportunities' in ai_insights:
            print(ai_insights['growth_opportunities'])
        
        print("\n=== AI-GENERATED STRATEGIC RECOMMENDATIONS ===")
        if 'strategic_recommendations' in ai_insights:
            print(ai_insights['strategic_recommendations'])
            
    else:
        # Fallback to traditional insights
        print("\nGenerating template-based insights...")
        analyzer.generate_insights()
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.create_visualizations()
    
    # Export comprehensive report
    print("\nExporting enhanced report...")
    report = analyzer.export_enhanced_report(include_openai=True)
    
    print("\nEnhanced Play Store Analysis Complete!")
    print("Check the generated visualizations and comprehensive JSON report.")
    
    return analyzer, report

# Additional utility functions for API integration with OpenAI enhancement
class AppStoreAPIIntegrator:
    """Enhanced class to handle App Store API integration with OpenAI insights"""
    
    def __init__(self, api_key, openai_api_key=None):
        self.api_key = api_key
        self.base_url = "https://app-store-scraper1.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "app-store-scraper1.p.rapidapi.com"
        }
        
        # Initialize OpenAI client
        self.openai_client = None
        if openai_api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                print("OpenAI integration enabled for App Store analysis")
            except Exception as e:
                print(f"OpenAI initialization failed: {e}")
    
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
        if isinstance(playstore_data, pd.DataFrame):
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
    
    def generate_cross_platform_insights(self, unified_data, max_tokens=1500):
        """Generate cross-platform insights using OpenAI"""
        if not self.openai_client:
            print("OpenAI not available for cross-platform insights")
            return "Cross-platform insights require OpenAI integration"
        
        # Prepare context for cross-platform analysis
        context = self._prepare_cross_platform_context(unified_data)
        
        prompt = f"""
        Analyze this cross-platform mobile app data (Google Play Store vs Apple App Store) and provide insights:

        CROSS-PLATFORM DATA:
        {json.dumps(context, indent=2)}

        Provide analysis on:
        1. Platform-specific performance differences
        2. Category preferences by platform
        3. Pricing strategy variations
        4. User engagement patterns across platforms
        5. Quality standards comparison
        6. Market opportunity differences
        7. Developer strategy recommendations for each platform
        8. Cross-platform optimization strategies

        Focus on actionable insights for multi-platform app development and marketing.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a cross-platform mobile app strategist with expertise in both iOS and Android markets, user behavior patterns, and platform-specific optimization."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating cross-platform insights: {e}")
            return "Cross-platform insights generation failed. Please check your OpenAI configuration."
    
    def _prepare_cross_platform_context(self, unified_data):
        """Prepare context for cross-platform analysis"""
        if unified_data.empty:
            return {}
        
        context = {
            'total_apps': len(unified_data),
            'platform_distribution': unified_data['platform'].value_counts().to_dict(),
        }
        
        # Platform-specific metrics
        for platform in unified_data['platform'].unique():
            platform_data = unified_data[unified_data['platform'] == platform]
            context[f'{platform.lower().replace(" ", "_")}_stats'] = {
                'app_count': len(platform_data),
                'avg_rating': float(platform_data['rating'].mean()) if 'rating' in platform_data.columns else 0,
                'avg_price': float(platform_data['price'].mean()) if 'price' in platform_data.columns else 0,
                'free_apps_percent': float((platform_data['price'] == 0).mean() * 100) if 'price' in platform_data.columns else 0,
                'top_categories': platform_data['category'].value_counts().head(5).to_dict() if 'category' in platform_data.columns else {}
            }
        
        return context

# Usage examples and comprehensive workflow
def run_complete_mobile_analysis(playstore_csv: str, openai_api_key: str = None, rapidapi_key: str = None):
    """
    Complete mobile app analysis workflow with OpenAI insights
    """
    print("Starting Complete Mobile App Market Analysis...")
    
    # Phase 1: Enhanced Play Store Analysis
    print("\n=== PHASE 1: GOOGLE PLAY STORE ANALYSIS ===")
    playstore_analyzer, playstore_report = run_enhanced_playstore_analysis(playstore_csv, openai_api_key)
    
    # Phase 2: App Store Integration (if API key provided)
    appstore_data = None
    if rapidapi_key:
        print("\n=== PHASE 2: APP STORE DATA INTEGRATION ===")
        api_integrator = AppStoreAPIIntegrator(rapidapi_key, openai_api_key)
        
        # Fetch sample App Store data for comparison
        categories = ['games', 'productivity', 'business', 'social', 'entertainment']
        appstore_apps = []
        
        for category in categories:
            print(f"Fetching {category} apps from App Store...")
            apps_data = api_integrator.search_apps(category, limit=20)
            if apps_data and 'results' in apps_data:
                appstore_apps.extend(apps_data['results'])
        
        if appstore_apps:
            appstore_data = appstore_apps
            print(f"Fetched {len(appstore_apps)} apps from App Store")
            
            # Create unified dataset
            unified_data = api_integrator.create_unified_schema(
                playstore_analyzer.cleaned_df, 
                appstore_data
            )
            
            # Generate cross-platform insights
            if api_integrator.openai_client:
                print("\n=== CROSS-PLATFORM AI INSIGHTS ===")
                cross_platform_insights = api_integrator.generate_cross_platform_insights(unified_data)
                print(cross_platform_insights)
    
    # Phase 3: Export comprehensive results
    print("\n=== PHASE 3: COMPREHENSIVE REPORTING ===")
    
    final_report = {
        'analysis_date': datetime.now().isoformat(),
        'playstore_analysis': playstore_report,
        'appstore_data_included': appstore_data is not None,
        'cross_platform_analysis': cross_platform_insights if rapidapi_key and openai_api_key else None,
        'recommendations': _generate_final_recommendations(playstore_analyzer, appstore_data)
    }
    
    # Export final comprehensive report
    with open('complete_mobile_analysis_report.json', 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print("Complete mobile analysis report exported to: complete_mobile_analysis_report.json")
    print("\nAnalysis Complete! Check all generated reports and visualizations.")
    
    return {
        'playstore_analyzer': playstore_analyzer,
        'playstore_report': playstore_report,
        'appstore_data': appstore_data,
        'final_report': final_report
    }

def _generate_final_recommendations(playstore_analyzer, appstore_data):
    """Generate final strategic recommendations"""
    recommendations = []
    
    if playstore_analyzer and playstore_analyzer.cleaned_df is not None:
        # Play Store specific recommendations
        if 'Category' in playstore_analyzer.cleaned_df.columns:
            category_counts = playstore_analyzer.cleaned_df['Category'].value_counts()
            underserved = category_counts[category_counts <= 50].head(3)
            
            for category in underserved.index:
                recommendations.append({
                    'type': 'Market Opportunity',
                    'platform': 'Google Play',
                    'recommendation': f'Consider entering {category} market - only {category_counts[category]} apps in dataset',
                    'priority': 'High' if category_counts[category] <= 20 else 'Medium'
                })
        
        # Quality recommendations
        if 'Rating' in playstore_analyzer.cleaned_df.columns:
            avg_rating = playstore_analyzer.cleaned_df['Rating'].mean()
            if avg_rating < 4.0:
                recommendations.append({
                    'type': 'Quality Focus',
                    'platform': 'Google Play',
                    'recommendation': f'Market average rating is {avg_rating:.2f}. Focus on quality to stand out (target 4.0+)',
                    'priority': 'High'
                })
    
    # Cross-platform recommendations
    if appstore_data:
        recommendations.append({
            'type': 'Cross-Platform Strategy',
            'platform': 'Both',
            'recommendation': 'Implement platform-specific optimization strategies based on user behavior differences',
            'priority': 'Medium'
        })
    
    return recommendations

# Usage instructions with examples
print("Enhanced Google Play Store Analytics with OpenAI Integration Ready!")
print("\nUsage Examples:")
print("# Basic analysis with OpenAI insights:")
print("analyzer, report = run_enhanced_playstore_analysis('googleplaystore.csv', 'your-openai-api-key')")
print("\n# Complete analysis with both platforms:")
print("results = run_complete_mobile_analysis('googleplaystore.csv', 'openai-key', 'rapidapi-key')")
print("\n# Generate specific AI insights:")
print("market_insights = analyzer.generate_openai_insights('market_analysis', max_tokens=1000)")
print("competitive_insights = analyzer.generate_openai_insights('competitive', max_tokens=1200)")
print("opportunities = analyzer.generate_openai_insights('opportunities', max_tokens=1500)")
