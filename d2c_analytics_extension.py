import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    print("⚠️ OpenAI library not installed. Install with: pip install openai")

class D2CAnalytics:
    """
    Enhanced D2C eCommerce Analytics Pipeline with OpenAI-powered insights
    for funnel optimization, SEO opportunities, and AI-generated creative content
    """
    
    def __init__(self, excel_file_path: str, openai_api_key: str = None):
        self.excel_file_path = excel_file_path
        self.raw_data = None
        self.processed_data = None
        self.funnel_metrics = None
        self.seo_insights = None
        self.openai_client = None
        
        # Initialize OpenAI client
        if openai_api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                print("✅ OpenAI integration enabled")
            except Exception as e:
                print(f"⚠️ OpenAI initialization failed: {e}")
        elif not openai_api_key:
            # Try to get from environment variable
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and OPENAI_AVAILABLE:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    print("✅ OpenAI integration enabled (from environment)")
                except Exception as e:
                    print(f"⚠️ OpenAI initialization failed: {e}")
            else:
                print("ℹ️ OpenAI API key not provided. AI insights will be template-based.")
        
    def load_d2c_data(self):
        """Load D2C eCommerce data from Excel file"""
        try:
            # Try to read Excel file - adjust sheet names as needed
            self.raw_data = pd.read_excel(self.excel_file_path, sheet_name=0)  # Assuming first sheet
            print(f"✅ D2C dataset loaded successfully!")
            print(f"Dataset shape: {self.raw_data.shape}")
            print(f"Columns: {list(self.raw_data.columns)}")
            print(f"\nSample data:")
            print(self.raw_data.head())
            return True
            
        except Exception as e:
            print(f"❌ Error loading D2C data: {e}")
            return False
    
    def preprocess_d2c_data(self):
        """Clean and preprocess D2C data"""
        if self.raw_data is None:
            print("Please load data first!")
            return False
        
        self.processed_data = self.raw_data.copy()
        
        # Clean and standardize column names
        self.processed_data.columns = [col.strip().lower().replace(' ', '_') for col in self.processed_data.columns]
        
        print("🧹 Data preprocessing completed")
        print(f"Processed columns: {list(self.processed_data.columns)}")
        
        # Basic data quality checks
        print(f"\nMissing values:")
        missing_data = self.processed_data.isnull().sum()
        print(missing_data[missing_data > 0])
        
        return True
    
    def analyze_funnel_performance(self):
        """
        Analyze D2C funnel metrics: Installs → Signups → First Purchase → Repeat Purchase
        Calculate CAC, ROAS, and retention patterns
        """
        print("=== FUNNEL PERFORMANCE ANALYSIS ===")
        
        if self.processed_data is None:
            print("Please preprocess data first!")
            return
        
        # Map common column variations to standard names
        column_mapping = self._map_columns()
        
        # Calculate funnel metrics
        funnel_data = []
        
        # Group by campaign or category for funnel analysis
        group_by_col = self._identify_grouping_column()
        
        if group_by_col:
            for group_name, group_data in self.processed_data.groupby(group_by_col):
                metrics = self._calculate_funnel_metrics(group_data, column_mapping)
                metrics['group'] = group_name
                funnel_data.append(metrics)
        
        self.funnel_metrics = pd.DataFrame(funnel_data)
        
        if not self.funnel_metrics.empty:
            print("📊 Funnel Performance Summary:")
            print(self.funnel_metrics.round(2))
            
            # Identify best and worst performing segments
            self._identify_performance_segments()
            
        return self.funnel_metrics
    
    def _map_columns(self):
        """Map various column name variations to standard metrics"""
        columns = self.processed_data.columns.tolist()
        
        mapping = {}
        
        # Common patterns for different metrics
        patterns = {
            'spend': ['spend', 'cost', 'budget', 'investment'],
            'impressions': ['impression', 'views', 'reach'],
            'clicks': ['click', 'ctr'],
            'installs': ['install', 'download', 'acquisition'],
            'signups': ['signup', 'registration', 'register'],
            'first_purchase': ['first_purchase', 'conversion', 'purchase'],
            'repeat_purchase': ['repeat', 'retention', 'return'],
            'revenue': ['revenue', 'income', 'sales'],
            'conversions': ['conversion', 'convert']
        }
        
        for metric, keywords in patterns.items():
            for col in columns:
                if any(keyword in col.lower() for keyword in keywords):
                    mapping[metric] = col
                    break
        
        print(f"📋 Column mapping identified: {mapping}")
        return mapping
    
    def _identify_grouping_column(self):
        """Identify the best column for grouping (campaign, category, etc.)"""
        columns = self.processed_data.columns.tolist()
        
        grouping_keywords = ['campaign', 'category', 'channel', 'source', 'segment']
        
        for col in columns:
            if any(keyword in col.lower() for keyword in grouping_keywords):
                return col
        
        # If no obvious grouping column, return None or first categorical column
        categorical_cols = self.processed_data.select_dtypes(include=['object']).columns
        return categorical_cols[0] if len(categorical_cols) > 0 else None
    
    def _calculate_funnel_metrics(self, data, column_mapping):
        """Calculate key funnel metrics for a data segment"""
        metrics = {}
        
        # Basic metrics
        metrics['total_spend'] = data[column_mapping.get('spend', data.columns[0])].sum() if 'spend' in column_mapping else 0
        metrics['total_impressions'] = data[column_mapping.get('impressions', data.columns[0])].sum() if 'impressions' in column_mapping else 0
        metrics['total_clicks'] = data[column_mapping.get('clicks', data.columns[0])].sum() if 'clicks' in column_mapping else 0
        metrics['total_installs'] = data[column_mapping.get('installs', data.columns[0])].sum() if 'installs' in column_mapping else 0
        metrics['total_signups'] = data[column_mapping.get('signups', data.columns[0])].sum() if 'signups' in column_mapping else 0
        metrics['total_first_purchases'] = data[column_mapping.get('first_purchase', data.columns[0])].sum() if 'first_purchase' in column_mapping else 0
        metrics['total_repeat_purchases'] = data[column_mapping.get('repeat_purchase', data.columns[0])].sum() if 'repeat_purchase' in column_mapping else 0
        metrics['total_revenue'] = data[column_mapping.get('revenue', data.columns[0])].sum() if 'revenue' in column_mapping else 0
        
        # Calculated metrics
        # CTR (Click-through rate)
        metrics['ctr'] = (metrics['total_clicks'] / metrics['total_impressions'] * 100) if metrics['total_impressions'] > 0 else 0
        
        # Install rate
        metrics['install_rate'] = (metrics['total_installs'] / metrics['total_clicks'] * 100) if metrics['total_clicks'] > 0 else 0
        
        # Signup conversion rate
        metrics['signup_rate'] = (metrics['total_signups'] / metrics['total_installs'] * 100) if metrics['total_installs'] > 0 else 0
        
        # First purchase conversion rate
        metrics['first_purchase_rate'] = (metrics['total_first_purchases'] / metrics['total_signups'] * 100) if metrics['total_signups'] > 0 else 0
        
        # Repeat purchase rate (retention)
        metrics['repeat_purchase_rate'] = (metrics['total_repeat_purchases'] / metrics['total_first_purchases'] * 100) if metrics['total_first_purchases'] > 0 else 0
        
        # CAC (Customer Acquisition Cost)
        metrics['cac'] = (metrics['total_spend'] / metrics['total_first_purchases']) if metrics['total_first_purchases'] > 0 else 0
        
        # ROAS (Return on Ad Spend)
        metrics['roas'] = (metrics['total_revenue'] / metrics['total_spend']) if metrics['total_spend'] > 0 else 0
        
        # LTV estimate (simplified)
        avg_order_value = metrics['total_revenue'] / metrics['total_first_purchases'] if metrics['total_first_purchases'] > 0 else 0
        estimated_ltv = avg_order_value * (1 + metrics['repeat_purchase_rate'] / 100)
        metrics['estimated_ltv'] = estimated_ltv
        
        # LTV:CAC ratio
        metrics['ltv_cac_ratio'] = (estimated_ltv / metrics['cac']) if metrics['cac'] > 0 else 0
        
        return metrics
    
    def _identify_performance_segments(self):
        """Identify best and worst performing segments"""
        if self.funnel_metrics.empty:
            return
        
        # Rank by ROAS
        top_roas = self.funnel_metrics.nlargest(3, 'roas')
        bottom_roas = self.funnel_metrics.nsmallest(3, 'roas')
        
        print(f"\n🏆 TOP 3 SEGMENTS BY ROAS:")
        print(top_roas[['group', 'roas', 'cac', 'ltv_cac_ratio']].round(2))
        
        print(f"\n⚠️ BOTTOM 3 SEGMENTS BY ROAS:")
        print(bottom_roas[['group', 'roas', 'cac', 'ltv_cac_ratio']].round(2))
        
        # Rank by LTV:CAC ratio
        top_ltv_cac = self.funnel_metrics.nlargest(3, 'ltv_cac_ratio')
        print(f"\n💰 TOP 3 SEGMENTS BY LTV:CAC RATIO:")
        print(top_ltv_cac[['group', 'ltv_cac_ratio', 'estimated_ltv', 'cac']].round(2))
    
    def analyze_seo_opportunities(self):
        """Analyze SEO metrics and identify growth opportunities"""
        print("=== SEO OPPORTUNITIES ANALYSIS ===")
        
        if self.processed_data is None:
            print("Please preprocess data first!")
            return
        
        # Map SEO-related columns
        seo_mapping = self._map_seo_columns()
        
        if not seo_mapping:
            print("❌ No SEO-related columns found in the dataset")
            return
        
        # Group by category for SEO analysis
        category_col = self._identify_category_column()
        
        if category_col:
            seo_data = []
            
            for category, group_data in self.processed_data.groupby(category_col):
                seo_metrics = self._calculate_seo_metrics(group_data, seo_mapping)
                seo_metrics['category'] = category
                seo_data.append(seo_metrics)
            
            self.seo_insights = pd.DataFrame(seo_data)
            
            if not self.seo_insights.empty:
                print("📈 SEO Performance by Category:")
                print(self.seo_insights.round(2))
                
                # Identify opportunities
                self._identify_seo_opportunities()
                
        return self.seo_insights
    
    def _map_seo_columns(self):
        """Map SEO-related columns"""
        columns = self.processed_data.columns.tolist()
        mapping = {}
        
        seo_patterns = {
            'search_volume': ['search_volume', 'volume', 'searches'],
            'avg_position': ['position', 'rank', 'ranking'],
            'conversion_rate': ['conversion_rate', 'cvr', 'convert'],
            'organic_traffic': ['organic', 'traffic', 'visits'],
            'clicks': ['clicks', 'click']
        }
        
        for metric, keywords in seo_patterns.items():
            for col in columns:
                if any(keyword in col.lower() for keyword in keywords):
                    mapping[metric] = col
                    break
        
        print(f"🔍 SEO column mapping: {mapping}")
        return mapping
    
    def _identify_category_column(self):
        """Find category column for SEO analysis"""
        columns = self.processed_data.columns.tolist()
        category_keywords = ['category', 'vertical', 'segment', 'product_type']
        
        for col in columns:
            if any(keyword in col.lower() for keyword in category_keywords):
                return col
        
        # Return first categorical column if no obvious category found
        categorical_cols = self.processed_data.select_dtypes(include=['object']).columns
        return categorical_cols[0] if len(categorical_cols) > 0 else None
    
    def _calculate_seo_metrics(self, data, seo_mapping):
        """Calculate SEO metrics for a category"""
        metrics = {}
        
        # Basic SEO metrics
        metrics['total_search_volume'] = data[seo_mapping.get('search_volume', data.columns[0])].sum() if 'search_volume' in seo_mapping else 0
        metrics['avg_position'] = data[seo_mapping.get('avg_position', data.columns[0])].mean() if 'avg_position' in seo_mapping else 0
        metrics['conversion_rate'] = data[seo_mapping.get('conversion_rate', data.columns[0])].mean() if 'conversion_rate' in seo_mapping else 0
        metrics['total_organic_traffic'] = data[seo_mapping.get('organic_traffic', data.columns[0])].sum() if 'organic_traffic' in seo_mapping else 0
        metrics['total_clicks'] = data[seo_mapping.get('clicks', data.columns[0])].sum() if 'clicks' in seo_mapping else 0
        
        # Calculated metrics
        # SEO opportunity score (high volume, poor position, good conversion rate)
        volume_score = min(metrics['total_search_volume'] / 10000, 1) if metrics['total_search_volume'] > 0 else 0
        position_opportunity = max(0, (50 - metrics['avg_position']) / 50) if metrics['avg_position'] > 0 else 0
        conversion_potential = metrics['conversion_rate'] / 100 if metrics['conversion_rate'] > 0 else 0
        
        metrics['seo_opportunity_score'] = (volume_score * 0.4 + position_opportunity * 0.4 + conversion_potential * 0.2)
        
        # Traffic potential (search volume * estimated CTR based on position)
        if metrics['avg_position'] > 0:
            estimated_ctr = self._estimate_ctr_by_position(metrics['avg_position'])
            metrics['traffic_potential'] = metrics['total_search_volume'] * estimated_ctr / 100
        else:
            metrics['traffic_potential'] = 0
        
        return metrics
    
    def _estimate_ctr_by_position(self, position):
        """Estimate CTR based on search position"""
        # Simplified CTR curve based on industry averages
        ctr_by_position = {
            1: 28, 2: 15, 3: 11, 4: 8, 5: 7,
            6: 5, 7: 4, 8: 3, 9: 2, 10: 2
        }
        
        if position <= 10:
            return ctr_by_position.get(int(position), 1)
        elif position <= 20:
            return 1
        else:
            return 0.5
    
    def _identify_seo_opportunities(self):
        """Identify top SEO growth opportunities"""
        if self.seo_insights.empty:
            return
        
        # Sort by opportunity score
        top_opportunities = self.seo_insights.nlargest(5, 'seo_opportunity_score')
        
        print(f"\n🎯 TOP 5 SEO OPPORTUNITIES:")
        print(top_opportunities[['category', 'seo_opportunity_score', 'total_search_volume', 'avg_position', 'conversion_rate']].round(2))
        
        # High volume, low performance categories
        high_volume_low_perf = self.seo_insights[
            (self.seo_insights['total_search_volume'] > self.seo_insights['total_search_volume'].median()) &
            (self.seo_insights['avg_position'] > 10)
        ].sort_values('total_search_volume', ascending=False)
        
        if not high_volume_low_perf.empty:
            print(f"\n📊 HIGH VOLUME, LOW PERFORMANCE CATEGORIES:")
            print(high_volume_low_perf[['category', 'total_search_volume', 'avg_position', 'traffic_potential']].round(2).head())
    
    def generate_openai_insights(self, insight_type: str = "comprehensive", max_tokens: int = 1500):
        """Generate advanced insights using OpenAI GPT models"""
        if not self.openai_client:
            print("⚠️ OpenAI client not available. Using template-based insights.")
            return self.generate_ai_creative_content()
        
        print("🤖 Generating OpenAI-powered insights...")
        
        # Prepare data context for OpenAI
        data_context = self._prepare_openai_context()
        
        insights = {}
        
        try:
            if insight_type in ["comprehensive", "strategic"]:
                insights['strategic_insights'] = self._generate_strategic_insights_openai(data_context, max_tokens)
            
            if insight_type in ["comprehensive", "creative"]:
                insights['creative_content'] = self._generate_creative_content_openai(data_context, max_tokens)
            
            if insight_type in ["comprehensive", "recommendations"]:
                insights['recommendations'] = self._generate_recommendations_openai(data_context, max_tokens)
            
            if insight_type in ["comprehensive", "market_analysis"]:
                insights['market_analysis'] = self._generate_market_analysis_openai(data_context, max_tokens)
            
            print("✅ OpenAI insights generated successfully!")
            return insights
            
        except Exception as e:
            print(f"❌ Error generating OpenAI insights: {e}")
            print("Falling back to template-based insights...")
            return self.generate_ai_creative_content()
    
    def _prepare_openai_context(self):
        """Prepare structured data context for OpenAI analysis"""
        context = {
            'dataset_summary': {
                'total_rows': len(self.processed_data) if self.processed_data is not None else 0,
                'columns': list(self.processed_data.columns) if self.processed_data is not None else [],
            },
            'funnel_performance': {},
            'seo_insights': {}
        }
        
        # Funnel metrics summary
        if self.funnel_metrics is not None and not self.funnel_metrics.empty:
            context['funnel_performance'] = {
                'segments_analyzed': len(self.funnel_metrics),
                'average_metrics': {
                    'ctr': float(self.funnel_metrics['ctr'].mean()),
                    'signup_rate': float(self.funnel_metrics['signup_rate'].mean()),
                    'first_purchase_rate': float(self.funnel_metrics['first_purchase_rate'].mean()),
                    'repeat_purchase_rate': float(self.funnel_metrics['repeat_purchase_rate'].mean()),
                    'cac': float(self.funnel_metrics['cac'].mean()),
                    'roas': float(self.funnel_metrics['roas'].mean()),
                    'ltv_cac_ratio': float(self.funnel_metrics['ltv_cac_ratio'].mean())
                },
                'best_performer': {
                    'segment': str(self.funnel_metrics.loc[self.funnel_metrics['roas'].idxmax(), 'group']),
                    'roas': float(self.funnel_metrics['roas'].max()),
                    'cac': float(self.funnel_metrics.loc[self.funnel_metrics['roas'].idxmax(), 'cac']),
                    'ltv_cac_ratio': float(self.funnel_metrics.loc[self.funnel_metrics['roas'].idxmax(), 'ltv_cac_ratio'])
                },
                'worst_performer': {
                    'segment': str(self.funnel_metrics.loc[self.funnel_metrics['roas'].idxmin(), 'group']),
                    'roas': float(self.funnel_metrics['roas'].min()),
                    'cac': float(self.funnel_metrics.loc[self.funnel_metrics['roas'].idxmin(), 'cac'])
                }
            }
        
        # SEO insights summary
        if self.seo_insights is not None and not self.seo_insights.empty:
            top_seo_opportunity = self.seo_insights.loc[self.seo_insights['seo_opportunity_score'].idxmax()]
            context['seo_insights'] = {
                'categories_analyzed': len(self.seo_insights),
                'total_search_volume': int(self.seo_insights['total_search_volume'].sum()),
                'average_position': float(self.seo_insights['avg_position'].mean()),
                'total_traffic_potential': int(self.seo_insights['traffic_potential'].sum()),
                'top_opportunity': {
                    'category': str(top_seo_opportunity['category']),
                    'opportunity_score': float(top_seo_opportunity['seo_opportunity_score']),
                    'search_volume': int(top_seo_opportunity['total_search_volume']),
                    'position': float(top_seo_opportunity['avg_position']),
                    'traffic_potential': int(top_seo_opportunity['traffic_potential'])
                }
            }
        
        return context
    
    def _generate_strategic_insights_openai(self, context, max_tokens):
        """Generate strategic insights using OpenAI"""
        prompt = f"""
        As a senior eCommerce strategist, analyze this D2C business performance data and provide strategic insights:

        BUSINESS PERFORMANCE DATA:
        {json.dumps(context, indent=2)}

        Please provide:
        1. Key performance insights and trends
        2. Critical business challenges identified
        3. Growth opportunities and market gaps
        4. Strategic recommendations for improvement
        5. Risk assessment and mitigation strategies

        Focus on actionable, data-driven insights that can drive business growth.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior eCommerce strategist with expertise in D2C business optimization, funnel analysis, and digital marketing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating strategic insights: {e}")
            return "Strategic insights generation failed. Please check your OpenAI configuration."
    
    def _generate_creative_content_openai(self, context, max_tokens):
        """Generate creative content using OpenAI"""
        best_segment = context.get('funnel_performance', {}).get('best_performer', {})
        top_seo = context.get('seo_insights', {}).get('top_opportunity', {})
        
        prompt = f"""
        Create high-converting marketing content based on this performance data:

        TOP PERFORMING SEGMENT: {best_segment.get('segment', 'N/A')}
        - ROAS: {best_segment.get('roas', 0):.2f}x
        - CAC: ${best_segment.get('cac', 0):.2f}
        - LTV:CAC Ratio: {best_segment.get('ltv_cac_ratio', 0):.2f}

        TOP SEO OPPORTUNITY: {top_seo.get('category', 'N/A')}
        - Search Volume: {top_seo.get('search_volume', 0):,} monthly
        - Current Position: {top_seo.get('position', 0):.1f}
        - Traffic Potential: {top_seo.get('traffic_potential', 0):,}

        Generate:
        1. 5 high-converting ad headlines
        2. 3 SEO-optimized meta descriptions (150-160 chars)
        3. 2 compelling product descriptions with social proof
        4. 3 email subject lines for retention campaigns
        5. 2 landing page headlines with value propositions

        Make content specific to the data and performance metrics provided.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a conversion copywriter specializing in D2C eCommerce with expertise in data-driven content creation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.8
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating creative content: {e}")
            return "Creative content generation failed. Please check your OpenAI configuration."
    
    def _generate_recommendations_openai(self, context, max_tokens):
        """Generate optimization recommendations using OpenAI"""
        avg_metrics = context.get('funnel_performance', {}).get('average_metrics', {})
        
        prompt = f"""
        Based on this D2C eCommerce funnel analysis, provide specific optimization recommendations:

        CURRENT PERFORMANCE METRICS:
        - Average CTR: {avg_metrics.get('ctr', 0):.2f}%
        - Signup Rate: {avg_metrics.get('signup_rate', 0):.2f}%
        - First Purchase Rate: {avg_metrics.get('first_purchase_rate', 0):.2f}%
        - Repeat Purchase Rate: {avg_metrics.get('repeat_purchase_rate', 0):.2f}%
        - Average CAC: ${avg_metrics.get('cac', 0):.2f}
        - Average ROAS: {avg_metrics.get('roas', 0):.2f}x
        - LTV:CAC Ratio: {avg_metrics.get('ltv_cac_ratio', 0):.2f}

        SEO ANALYSIS:
        - Total Traffic Potential: {context.get('seo_insights', {}).get('total_traffic_potential', 0):,}
        - Average Position: {context.get('seo_insights', {}).get('average_position', 0):.1f}

        Provide prioritized recommendations for:
        1. Funnel optimization (specific tactics for each stage)
        2. Customer acquisition cost reduction
        3. Retention and LTV improvement
        4. SEO strategy and content optimization
        5. Budget reallocation suggestions
        6. Technology and tool recommendations

        Include expected impact estimates and implementation timelines.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a growth marketing expert specializing in D2C eCommerce optimization with deep expertise in conversion rate optimization, customer acquisition, and retention strategies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.6
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return "Recommendations generation failed. Please check your OpenAI configuration."
    
    def _generate_market_analysis_openai(self, context, max_tokens):
        """Generate market analysis using OpenAI"""
        prompt = f"""
        Conduct a comprehensive market analysis based on this D2C eCommerce data:

        BUSINESS CONTEXT:
        {json.dumps(context, indent=2)}

        Provide analysis on:
        1. Market positioning and competitive landscape insights
        2. Customer behavior patterns and segment analysis
        3. Industry benchmarking (compare to typical D2C metrics)
        4. Market trends and opportunities
        5. Seasonal patterns and cyclical behaviors
        6. Customer lifetime value optimization opportunities
        7. Market expansion recommendations
        8. Pricing strategy insights

        Focus on market-level insights that inform strategic decisions.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a market research analyst specializing in D2C eCommerce markets with expertise in customer behavior analysis, competitive intelligence, and market trend identification."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating market analysis: {e}")
            return "Market analysis generation failed. Please check your OpenAI configuration."
    
    def generate_ai_creative_content(self):
        """Generate AI-powered creative outputs based on insights (fallback method)"""
        print("=== AI-POWERED CREATIVE GENERATION ===")
        
        if self.funnel_metrics is None or self.seo_insights is None:
            print("Please run funnel and SEO analysis first!")
            return
        
        creative_outputs = {}
        
        # 1. Ad Headlines based on best performing segments
        creative_outputs['ad_headlines'] = self._generate_ad_headlines()
        
        # 2. SEO Meta Descriptions based on top opportunities
        creative_outputs['meta_descriptions'] = self._generate_meta_descriptions()
        
        # 3. Product Description Text based on performance data
        creative_outputs['product_descriptions'] = self._generate_product_descriptions()
        
        return creative_outputs
    
    def _generate_ad_headlines(self):
        """Generate ad headlines based on top performing segments"""
        headlines = []
        
        if not self.funnel_metrics.empty:
            # Get top performing segment
            top_segment = self.funnel_metrics.loc[self.funnel_metrics['roas'].idxmax()]
            
            templates = [
                f"Get {top_segment['first_purchase_rate']:.0f}% More Conversions with {top_segment['group']}",
                f"Proven {top_segment['group']} Strategy - {top_segment['roas']:.1f}x ROAS",
                f"Transform Your {top_segment['group']} Results in 30 Days",
                f"Join {top_segment['total_first_purchases']:.0f}+ Happy {top_segment['group']} Customers"
            ]
            
            headlines.extend(templates)
        
        return headlines[:3]  # Return top 3
    
    def _generate_meta_descriptions(self):
        """Generate SEO meta descriptions for top opportunity categories"""
        meta_descriptions = []
        
        if not self.seo_insights.empty:
            top_seo_categories = self.seo_insights.nlargest(3, 'seo_opportunity_score')
            
            for _, category_data in top_seo_categories.iterrows():
                category = category_data['category']
                volume = category_data['total_search_volume']
                
                templates = [
                    f"Discover premium {category} solutions trusted by thousands. Get expert advice, competitive prices, and fast delivery. Search volume: {volume:,.0f} monthly searches.",
                    f"Shop the best {category} collection online. Compare top brands, read reviews, and save up to 50%. Free shipping available on all {category} orders.",
                    f"Your ultimate {category} destination. Professional-grade products, expert support, and satisfaction guarantee. Join {volume:,.0f}+ monthly shoppers."
                ]
                
                meta_descriptions.append({
                    'category': category,
                    'meta_description': templates[0]  # Use first template
                })
        
        return meta_descriptions
    
    def _generate_product_descriptions(self):
        """Generate product descriptions based on conversion data"""
        descriptions = []
        
        if not self.funnel_metrics.empty:
            # Focus on categories with good conversion rates
            high_converting = self.funnel_metrics[
                self.funnel_metrics['first_purchase_rate'] > self.funnel_metrics['first_purchase_rate'].median()
            ]
            
            for _, segment_data in high_converting.head(2).iterrows():
                category = segment_data['group']
                conversion_rate = segment_data['first_purchase_rate']
                
                description = f"""
**Transform Your {category} Experience**

Join the {segment_data['total_first_purchases']:.0f}+ satisfied customers who've discovered the difference quality makes. 

✅ {conversion_rate:.0f}% customer satisfaction rate
✅ Premium quality guaranteed  
✅ Fast, reliable delivery
✅ 30-day money-back guarantee

*"This {category} solution exceeded all my expectations. The quality and service are unmatched!"* - Verified Customer

**Limited Time: Free shipping on orders over $50**

Order now and experience the difference that's made us the #1 choice for {category} enthusiasts.
                """.strip()
                
                descriptions.append({
                    'category': category,
                    'product_description': description
                })
        
        return descriptions
    
    def create_d2c_dashboard(self):
        """Create comprehensive D2C analytics dashboard"""
        if self.funnel_metrics is None or self.seo_insights is None:
            print("Please run all analyses first!")
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Funnel conversion rates
        ax1 = fig.add_subplot(gs[0, 0])
        if not self.funnel_metrics.empty:
            conversion_metrics = ['ctr', 'install_rate', 'signup_rate', 'first_purchase_rate', 'repeat_purchase_rate']
            avg_conversions = [self.funnel_metrics[metric].mean() for metric in conversion_metrics]
            
            ax1.bar(range(len(conversion_metrics)), avg_conversions, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            ax1.set_xticks(range(len(conversion_metrics)))
            ax1.set_xticklabels([m.replace('_', '\n') for m in conversion_metrics], rotation=0, fontsize=8)
            ax1.set_title('Average Funnel Conversion Rates (%)', fontweight='bold')
            ax1.set_ylabel('Conversion Rate (%)')
        
        # 2. ROAS by segment
        ax2 = fig.add_subplot(gs[0, 1])
        if not self.funnel_metrics.empty:
            roas_data = self.funnel_metrics.nlargest(10, 'roas')
            ax2.barh(range(len(roas_data)), roas_data['roas'])
            ax2.set_yticks(range(len(roas_data)))
            ax2.set_yticklabels(roas_data['group'], fontsize=8)
            ax2.set_title('ROAS by Segment', fontweight='bold')
            ax2.set_xlabel('ROAS (Return on Ad Spend)')
        
        # 3. CAC vs LTV
        ax3 = fig.add_subplot(gs[0, 2])
        if not self.funnel_metrics.empty:
            ax3.scatter(self.funnel_metrics['cac'], self.funnel_metrics['estimated_ltv'], 
                       s=100, alpha=0.6, c=self.funnel_metrics['roas'], cmap='viridis')
            ax3.plot([0, max(self.funnel_metrics['cac'])], [0, max(self.funnel_metrics['cac'])], 
                    'r--', alpha=0.5, label='Break-even line')
            ax3.set_xlabel('Customer Acquisition Cost (CAC)')
            ax3.set_ylabel('Estimated Lifetime Value (LTV)')
            ax3.set_title('CAC vs LTV Analysis', fontweight='bold')
            ax3.legend()
            plt.colorbar(ax3.collections[0], ax=ax3, label='ROAS')
        
        # 4. SEO Opportunity Score
        ax4 = fig.add_subplot(gs[1, 0])
        if not self.seo_insights.empty:
            top_seo = self.seo_insights.nlargest(8, 'seo_opportunity_score')
            ax4.barh(range(len(top_seo)), top_seo['seo_opportunity_score'])
            ax4.set_yticks(range(len(top_seo)))
            ax4.set_yticklabels(top_seo['category'], fontsize=8)
            ax4.set_title('SEO Opportunity Score by Category', fontweight='bold')
            ax4.set_xlabel('Opportunity Score')
        
        # 5. Search Volume vs Position
        ax5 = fig.add_subplot(gs[1, 1])
        if not self.seo_insights.empty:
            scatter = ax5.scatter(self.seo_insights['avg_position'], self.seo_insights['total_search_volume'],
                                s=self.seo_insights['conversion_rate'] * 10, alpha=0.6,
                                c=self.seo_insights['seo_opportunity_score'], cmap='RdYlBu_r')
            ax5.set_xlabel('Average Position')
            ax5.set_ylabel('Total Search Volume')
            ax5.set_title('SEO Position vs Search Volume\n(Size = Conversion Rate)', fontweight='bold')
            plt.colorbar(scatter, ax=ax5, label='Opportunity Score')
        
        # 6. Traffic Potential
        ax6 = fig.add_subplot(gs[1, 2])
        if not self.seo_insights.empty:
            traffic_potential = self.seo_insights.nlargest(8, 'traffic_potential')
            ax6.pie(traffic_potential['traffic_potential'], labels=traffic_potential['category'], 
                   autopct='%1.1f%%', startangle=90)
            ax6.set_title('Traffic Potential Distribution', fontweight='bold')
        
        # 7. Funnel Performance Heatmap
        ax7 = fig.add_subplot(gs[2, :])
        if not self.funnel_metrics.empty:
            # Create heatmap data
            heatmap_metrics = ['ctr', 'install_rate', 'signup_rate', 'first_purchase_rate', 'repeat_purchase_rate', 'roas']
            heatmap_data = self.funnel_metrics[heatmap_metrics + ['group']].set_index('group')
            
            sns.heatmap(heatmap_data.head(10).T, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax7)
            ax7.set_title('Funnel Performance Heatmap (Top 10 Segments)', fontweight='bold')
            ax7.set_ylabel('Metrics')
        
        # 8. ROI Analysis
        ax8 = fig.add_subplot(gs[3, :2])
        if not self.funnel_metrics.empty:
            # Calculate ROI metrics
            roi_data = self.funnel_metrics.copy()
            roi_data['profit_margin'] = roi_data['total_revenue'] - roi_data['total_spend']
            roi_data = roi_data.nlargest(10, 'profit_margin')
            
            x = np.arange(len(roi_data))
            ax8.bar(x - 0.2, roi_data['total_spend'], 0.4, label='Spend', alpha=0.8)
            ax8.bar(x + 0.2, roi_data['total_revenue'], 0.4, label='Revenue', alpha=0.8)
            
            ax8.set_xlabel('Segments (Top 10 by Profit)')
            ax8.set_ylabel('Amount ($)')
            ax8.set_title('Revenue vs Spend Analysis', fontweight='bold')
            ax8.set_xticks(x)
            ax8.set_xticklabels(roi_data['group'], rotation=45, ha='right', fontsize=8)
            ax8.legend()
        
        # 9. Key Metrics Summary
        ax9 = fig.add_subplot(gs[3, 2])
        ax9.axis('off')
        
        # Calculate summary metrics
        if not self.funnel_metrics.empty:
            total_spend = self.funnel_metrics['total_spend'].sum()
            total_revenue = self.funnel_metrics['total_revenue'].sum()
            avg_roas = self.funnel_metrics['roas'].mean()
            avg_cac = self.funnel_metrics['cac'].mean()
            avg_ltv = self.funnel_metrics['estimated_ltv'].mean()
            
            summary_text = f"""
📊 PERFORMANCE SUMMARY
                
💰 Total Spend: ${total_spend:,.0f}
💵 Total Revenue: ${total_revenue:,.0f}
📈 Average ROAS: {avg_roas:.2f}x
🎯 Average CAC: ${avg_cac:.2f}
💎 Average LTV: ${avg_ltv:.2f}
📊 LTV:CAC Ratio: {avg_ltv/avg_cac:.2f}

🏆 Best Segment: {self.funnel_metrics.loc[self.funnel_metrics['roas'].idxmax(), 'group']}
⚠️ Needs Attention: {self.funnel_metrics.loc[self.funnel_metrics['roas'].idxmin(), 'group']}
            """.strip()
            
            ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('D2C eCommerce Performance Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def export_enhanced_report(self, filename: str = 'd2c_enhanced_insights_report.json', include_openai: bool = True):
        """Export comprehensive D2C insights report with OpenAI insights"""
        
        # Generate OpenAI insights if available
        openai_insights = {}
        if include_openai and self.openai_client:
            try:
                openai_insights = self.generate_openai_insights("comprehensive", max_tokens=2000)
            except Exception as e:
                print(f"Warning: Could not generate OpenAI insights: {e}")
        
        # Generate fallback creative content
        creative_content = self.generate_ai_creative_content()
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'openai_enabled': self.openai_client is not None,
            'funnel_analysis': {
                'total_segments': len(self.funnel_metrics) if self.funnel_metrics is not None else 0,
                'best_performing_segment': self._get_best_segment_info(),
                'average_metrics': self._get_average_funnel_metrics(),
                'recommendations': self._get_funnel_recommendations()
            },
            'seo_analysis': {
                'total_categories': len(self.seo_insights) if self.seo_insights is not None else 0,
                'top_opportunities': self._get_top_seo_opportunities(),
                'traffic_potential': self._get_traffic_potential_summary(),
                'recommendations': self._get_seo_recommendations()
            },
            'creative_content': creative_content,
            'openai_insights': openai_insights,
            'strategic_recommendations': self._get_strategic_recommendations()
        }
        
        # Export to JSON
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Enhanced D2C insights report exported to: {filename}")
        return report
    
    def _get_best_segment_info(self):
        """Get information about the best performing segment"""
        if self.funnel_metrics is None or self.funnel_metrics.empty:
            return {}
        
        best_segment = self.funnel_metrics.loc[self.funnel_metrics['roas'].idxmax()]
        return {
            'segment_name': best_segment['group'],
            'roas': float(best_segment['roas']),
            'cac': float(best_segment['cac']),
            'ltv_cac_ratio': float(best_segment['ltv_cac_ratio']),
            'conversion_rate': float(best_segment['first_purchase_rate'])
        }
    
    def _get_average_funnel_metrics(self):
        """Get average funnel metrics"""
        if self.funnel_metrics is None or self.funnel_metrics.empty:
            return {}
        
        return {
            'avg_ctr': float(self.funnel_metrics['ctr'].mean()),
            'avg_install_rate': float(self.funnel_metrics['install_rate'].mean()),
            'avg_signup_rate': float(self.funnel_metrics['signup_rate'].mean()),
            'avg_first_purchase_rate': float(self.funnel_metrics['first_purchase_rate'].mean()),
            'avg_repeat_purchase_rate': float(self.funnel_metrics['repeat_purchase_rate'].mean()),
            'avg_roas': float(self.funnel_metrics['roas'].mean()),
            'avg_cac': float(self.funnel_metrics['cac'].mean())
        }
    
    def _get_top_seo_opportunities(self):
        """Get top SEO opportunities"""
        if self.seo_insights is None or self.seo_insights.empty:
            return []
        
        top_3 = self.seo_insights.nlargest(3, 'seo_opportunity_score')
        return [
            {
                'category': row['category'],
                'opportunity_score': float(row['seo_opportunity_score']),
                'search_volume': int(row['total_search_volume']),
                'avg_position': float(row['avg_position']),
                'traffic_potential': int(row['traffic_potential'])
            }
            for _, row in top_3.iterrows()
        ]
    
    def _get_traffic_potential_summary(self):
        """Get traffic potential summary"""
        if self.seo_insights is None or self.seo_insights.empty:
            return {}
        
        return {
            'total_potential_traffic': int(self.seo_insights['traffic_potential'].sum()),
            'avg_search_volume': int(self.seo_insights['total_search_volume'].mean()),
            'categories_analyzed': len(self.seo_insights)
        }
    
    def _get_funnel_recommendations(self):
        """Generate funnel optimization recommendations"""
        if self.funnel_metrics is None or self.funnel_metrics.empty:
            return []
        
        recommendations = []
        
        # Low CTR segments
        low_ctr = self.funnel_metrics[self.funnel_metrics['ctr'] < self.funnel_metrics['ctr'].median()]
        if not low_ctr.empty:
            recommendations.append({
                'type': 'CTR Optimization',
                'priority': 'High',
                'recommendation': f"Optimize ad creative for {len(low_ctr)} segments with below-average CTR. Focus on compelling headlines and visuals."
            })
        
        # High CAC segments
        high_cac = self.funnel_metrics[self.funnel_metrics['cac'] > self.funnel_metrics['cac'].quantile(0.75)]
        if not high_cac.empty:
            recommendations.append({
                'type': 'CAC Optimization',
                'priority': 'Medium',
                'recommendation': f"Review targeting and bidding strategy for {len(high_cac)} high-CAC segments. Consider audience refinement."
            })
        
        # Low retention segments
        low_retention = self.funnel_metrics[self.funnel_metrics['repeat_purchase_rate'] < 20]
        if not low_retention.empty:
            recommendations.append({
                'type': 'Retention Improvement',
                'priority': 'High',
                'recommendation': f"Implement retention campaigns for {len(low_retention)} segments with low repeat purchase rates. Consider email marketing and loyalty programs."
            })
        
        return recommendations
    
    def _get_seo_recommendations(self):
        """Generate SEO optimization recommendations"""
        if self.seo_insights is None or self.seo_insights.empty:
            return []
        
        recommendations = []
        
        # High opportunity categories
        top_opportunities = self.seo_insights.nlargest(3, 'seo_opportunity_score')
        for _, opp in top_opportunities.iterrows():
            recommendations.append({
                'type': 'SEO Priority',
                'category': opp['category'],
                'priority': 'High',
                'recommendation': f"Focus SEO efforts on '{opp['category']}' - high search volume ({opp['total_search_volume']:,.0f}) with improvement potential (position {opp['avg_position']:.1f})"
            })
        
        # Content gaps
        poor_positions = self.seo_insights[self.seo_insights['avg_position'] > 20]
        if not poor_positions.empty:
            recommendations.append({
                'type': 'Content Optimization',
                'priority': 'Medium',
                'recommendation': f"Create targeted content for {len(poor_positions)} categories ranking below position 20. Focus on keyword optimization and content quality."
            })
        
        return recommendations
    
    def _get_strategic_recommendations(self):
        """Generate high-level strategic recommendations"""
        recommendations = []
        
        # Budget reallocation
        if self.funnel_metrics is not None and not self.funnel_metrics.empty:
            best_roas_segments = self.funnel_metrics.nlargest(3, 'roas')
            recommendations.append({
                'type': 'Budget Optimization',
                'recommendation': f"Reallocate budget to top-performing segments: {', '.join(best_roas_segments['group'].tolist())}. These show {best_roas_segments['roas'].mean():.2f}x average ROAS."
            })
        
        # Market expansion
        if self.seo_insights is not None and not self.seo_insights.empty:
            high_potential = self.seo_insights.nlargest(2, 'traffic_potential')
            recommendations.append({
                'type': 'Market Expansion',
                'recommendation': f"Consider expanding into {', '.join(high_potential['category'].tolist())} categories based on high traffic potential and SEO opportunities."
            })
        
        # Technology stack
        recommendations.append({
            'type': 'Technology Enhancement',
            'recommendation': "Implement advanced attribution modeling to better track customer journeys across touchpoints and optimize funnel performance."
        })
        
        return recommendations

# Enhanced main execution function with OpenAI integration
def run_enhanced_d2c_analysis(excel_file_path: str, openai_api_key: str = None):
    """
    Complete D2C analysis workflow with OpenAI-powered insights
    """
    print("🚀 Starting Enhanced D2C eCommerce Analysis Pipeline with AI Insights...")
    
    # Initialize analyzer with OpenAI
    analyzer = D2CAnalytics(excel_file_path, openai_api_key)
    
    # Load and preprocess data
    print("\n📊 Loading D2C dataset...")
    if not analyzer.load_d2c_data():
        return None
    
    print("\n🧹 Preprocessing data...")
    if not analyzer.preprocess_d2c_data():
        return None
    
    # Run analyses
    print("\n📈 Analyzing funnel performance...")
    analyzer.analyze_funnel_performance()
    
    print("\n🔍 Analyzing SEO opportunities...")
    analyzer.analyze_seo_opportunities()
    
    # Generate AI-powered insights
    print("\n🤖 Generating AI-powered insights...")
    if analyzer.openai_client:
        ai_insights = analyzer.generate_openai_insights("comprehensive", max_tokens=2000)
        
        print("\n=== AI-GENERATED STRATEGIC INSIGHTS ===")
        if 'strategic_insights' in ai_insights:
            print(ai_insights['strategic_insights'])
        
        print("\n=== AI-GENERATED CREATIVE CONTENT ===")
        if 'creative_content' in ai_insights:
            print(ai_insights['creative_content'])
        
        print("\n=== AI-GENERATED RECOMMENDATIONS ===")
        if 'recommendations' in ai_insights:
            print(ai_insights['recommendations'])
            
    else:
        # Fallback to template-based creative content
        print("\n🎨 Generating template-based creative content...")
        creative_outputs = analyzer.generate_ai_creative_content()
        
        print("\n📢 AD HEADLINES:")
        for i, headline in enumerate(creative_outputs.get('ad_headlines', []), 1):
            print(f"{i}. {headline}")
        
        print("\n📝 SEO META DESCRIPTIONS:")
        for meta in creative_outputs.get('meta_descriptions', []):
            print(f"\nCategory: {meta['category']}")
            print(f"Meta Description: {meta['meta_description']}")
        
        print("\n📦 PRODUCT DESCRIPTIONS:")
        for prod in creative_outputs.get('product_descriptions', []):
            print(f"\n--- {prod['category']} ---")
            print(prod['product_description'])
    
    # Create dashboard
    print("\n📊 Creating enhanced D2C dashboard...")
    analyzer.create_d2c_dashboard()
    
    # Export comprehensive report
    print("\n📄 Exporting enhanced insights report...")
    report = analyzer.export_enhanced_report(include_openai=True)
    
    print("\n✅ Enhanced D2C Analysis Complete!")
    print("Check the generated dashboard, AI insights, and comprehensive JSON report.")
    
    return analyzer, report

# Usage examples:
"""
# Run enhanced analysis with OpenAI
analyzer, report = run_enhanced_d2c_analysis('d2c_data.xlsx', 'your-openai-api-key')

# Run with environment variable
import os
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'
analyzer, report = run_enhanced_d2c_analysis('d2c_data.xlsx')

# Generate specific insights
if analyzer.openai_client:
    strategic_insights = analyzer.generate_openai_insights('strategic', max_tokens=1000)
    creative_content = analyzer.generate_openai_insights('creative', max_tokens=1500)
    recommendations = analyzer.generate_openai_insights('recommendations', max_tokens=1200)
"""

print("🤖 Enhanced D2C Analytics with OpenAI Integration Ready!")
print("Use run_enhanced_d2c_analysis() to start analysis with AI-powered insights.")
