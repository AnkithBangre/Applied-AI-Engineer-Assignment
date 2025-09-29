import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class D2CAnalytics:
    """
    D2C eCommerce Analytics Pipeline for funnel insights, SEO opportunities, 
    and AI-powered creative generation
    """
    
    def __init__(self, excel_file_path: str):
        self.excel_file_path = excel_file_path
        self.raw_data = None
        self.processed_data = None
        self.funnel_metrics = None
        self.seo_insights = None
        
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
    
    def generate_ai_creative_content(self):
        """Generate AI-powered creative outputs based on insights"""
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
    
    def export_d2c_report(self, filename: str = 'd2c_insights_report.json'):
        """Export comprehensive D2C insights report"""
        creative_content = self.generate_ai_creative_content()
        
        report = {
            'report_generated': datetime.now().isoformat(),
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
            'strategic_recommendations': self._get_strategic_recommendations()
        }
        
        # Export to JSON
        import json
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 D2C insights report exported to: {filename}")
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

# Main execution function
def run_d2c_analysis(excel_file_path: str):
    """
    Complete D2C analysis workflow
    """
    print("🚀 Starting D2C eCommerce Analysis Pipeline...")
    
    # Initialize analyzer
    analyzer = D2CAnalytics(excel_file_path)
    
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
    
    # Generate creative content
    print("\n🎨 Generating AI-powered creative content...")
    creative_outputs = analyzer.generate_ai_creative_content()
    
    # Display creative outputs
    print("\n=== AI-GENERATED CREATIVE CONTENT ===")
    
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
    print("\n📊 Creating D2C dashboard...")
    analyzer.create_d2c_dashboard()
    
    # Export report
    print("\n📄 Exporting comprehensive report...")
    report = analyzer.export_d2c_report()
    
    print("\n✅ D2C Analysis Complete!")
    print("Check the generated dashboard, creative content, and JSON report for insights.")
    
    return analyzer, report

# Example usage:
"""
# Run D2C analysis
analyzer, report = run_d2c_analysis('d2c_ecommerce_data.xlsx')

# Access specific insights
if analyzer.funnel_metrics is not None:
    print("Top performing segment:", analyzer.funnel_metrics.loc[analyzer.funnel_metrics['roas'].idxmax(), 'group'])

if analyzer.seo_insights is not None:
    print("Top SEO opportunity:", analyzer.seo_insights.loc[analyzer.seo_insights['seo_opportunity_score'].idxmax(), 'category'])
"""

class IntegratedAnalyticsPipeline:
    """
    Integrated pipeline combining Mobile App Analytics with D2C eCommerce Analytics
    """
    
    def __init__(self, rapidapi_key: str = None):
        self.mobile_analyzer = None
        self.d2c_analyzer = None
        self.rapidapi_key = rapidapi_key
        self.cross_platform_insights = {}
    
    def run_complete_pipeline(self, playstore_csv: str, d2c_excel: str, fetch_appstore: bool = False):
        """Run complete analytics pipeline across mobile and D2C"""
        print("🚀 INTEGRATED ANALYTICS PIPELINE STARTING...")
        print("=" * 60)
        
        # Phase 1-4: Mobile App Analytics
        print("\n📱 PHASE 1-4: MOBILE APP ANALYTICS")
        print("-" * 40)
        
        from .unified_mobile_analytics import UnifiedMobileAnalytics
        self.mobile_analyzer = UnifiedMobileAnalytics(self.rapidapi_key)
        
        # Load Play Store data
        if self.mobile_analyzer.load_playstore_data(playstore_csv):
            # Fetch App Store data if requested
            if fetch_appstore and self.rapidapi_key:
                self.mobile_analyzer.fetch_appstore_data()
            
            # Create unified dataset and run analyses
            self.mobile_analyzer.create_unified_dataset()
            self.mobile_analyzer.analyze_market_comparison()
            self.mobile_analyzer.analyze_category_performance()
            self.mobile_analyzer.identify_market_opportunities()
            self.mobile_analyzer.create_comprehensive_dashboard()
            mobile_report = self.mobile_analyzer.export_insights_report('mobile_insights_report.json')
        
        # Phase 5: D2C eCommerce Analytics
        print("\n🛒 PHASE 5: D2C ECOMMERCE ANALYTICS")
        print("-" * 40)
        
        self.d2c_analyzer = D2CAnalytics(d2c_excel)
        d2c_analyzer, d2c_report = run_d2c_analysis(d2c_excel)
        self.d2c_analyzer = d2c_analyzer
        
        # Cross-platform insights
        print("\n🔄 GENERATING CROSS-PLATFORM INSIGHTS")
        print("-" * 40)
        self._generate_cross_platform_insights()
        
        # Final integrated report
        print("\n📊 CREATING INTEGRATED REPORT")
        print("-" * 40)
        integrated_report = self._create_integrated_report(mobile_report, d2c_report)
        
        print("\n🎉 INTEGRATED ANALYSIS COMPLETE!")
        print("=" * 60)
        
        return {
            'mobile_analyzer': self.mobile_analyzer,
            'd2c_analyzer': self.d2c_analyzer,
            'integrated_report': integrated_report,
            'cross_platform_insights': self.cross_platform_insights
        }
    
    def _generate_cross_platform_insights(self):
        """Generate insights that span both mobile and D2C platforms"""
        
        insights = {
            'customer_journey_optimization': [],
            'unified_marketing_strategy': [],
            'technology_recommendations': [],
            'budget_allocation_strategy': []
        }
        
        # Customer Journey Optimization
        if self.mobile_analyzer and self.mobile_analyzer.unified_data is not None:
            mobile_categories = set(self.mobile_analyzer.unified_data['category'].unique())
            
            if self.d2c_analyzer and self.d2c_analyzer.processed_data is not None:
                # Find category overlaps
                category_col = self.d2c_analyzer._identify_category_column()
                if category_col:
                    d2c_categories = set(self.d2c_analyzer.processed_data[category_col].unique())
                    overlap_categories = mobile_categories.intersection(d2c_categories)
                    
                    if overlap_categories:
                        insights['customer_journey_optimization'].append({
                            'insight': 'Mobile-to-D2C Journey Opportunity',
                            'categories': list(overlap_categories),
                            'recommendation': f"Create unified customer journeys for {len(overlap_categories)} overlapping categories. Use mobile apps for discovery and D2C for conversion optimization."
                        })
        
        # Unified Marketing Strategy
        if self.mobile_analyzer and self.d2c_analyzer:
            if (hasattr(self.mobile_analyzer, 'unified_data') and 
                hasattr(self.d2c_analyzer, 'funnel_metrics') and
                self.d2c_analyzer.funnel_metrics is not None):
                
                # Compare conversion rates
                mobile_engagement = "High" if self.mobile_analyzer.unified_data['review_count'].mean() > 1000 else "Medium"
                d2c_conversion = "High" if self.d2c_analyzer.funnel_metrics['first_purchase_rate'].mean() > 5 else "Medium"
                
                insights['unified_marketing_strategy'].append({
                    'mobile_engagement': mobile_engagement,
                    'd2c_conversion_performance': d2c_conversion,
                    'recommendation': self._get_unified_strategy_recommendation(mobile_engagement, d2c_conversion)
                })
        
        # Technology Recommendations
        insights['technology_recommendations'].extend([
            {
                'area': 'Attribution Modeling',
                'recommendation': 'Implement cross-platform attribution to track user journeys from mobile discovery to D2C conversion'
            },
            {
                'area': 'Customer Data Platform',
                'recommendation': 'Unify mobile app analytics with D2C ecommerce data for 360-degree customer view'
            },
            {
                'area': 'Marketing Automation',
                'recommendation': 'Create automated workflows that trigger D2C campaigns based on mobile app behavior'
            }
        ])
        
        # Budget Allocation Strategy
        if (self.mobile_analyzer and hasattr(self.mobile_analyzer, 'unified_data') and
            self.d2c_analyzer and self.d2c_analyzer.funnel_metrics is not None):
            
            mobile_free_percentage = (self.mobile_analyzer.unified_data['type'] == 'Free').mean() * 100
            d2c_avg_cac = self.d2c_analyzer.funnel_metrics['cac'].mean()
            d2c_avg_roas = self.d2c_analyzer.funnel_metrics['roas'].mean()
            
            insights['budget_allocation_strategy'].append({
                'mobile_monetization_model': f"{mobile_free_percentage:.1f}% free apps suggest freemium focus",
                'd2c_efficiency': f"Average CAC: ${d2c_avg_cac:.2f}, ROAS: {d2c_avg_roas:.2f}x",
                'recommendation': self._get_budget_allocation_recommendation(mobile_free_percentage, d2c_avg_roas)
            })
        
        self.cross_platform_insights = insights
        
        # Print insights
        print("🔍 Cross-Platform Insights Generated:")
        for category, insight_list in insights.items():
            if insight_list:
                print(f"\n{category.replace('_', ' ').title()}:")
                for insight in insight_list:
                    if isinstance(insight, dict) and 'recommendation' in insight:
                        print(f"  • {insight['recommendation']}")
    
    def _get_unified_strategy_recommendation(self, mobile_engagement, d2c_conversion):
        """Get recommendation based on mobile engagement and D2C conversion performance"""
        strategies = {
            ('High', 'High'): "Leverage high mobile engagement to drive D2C conversions. Implement cross-platform retargeting and loyalty programs.",
            ('High', 'Medium'): "Focus on optimizing D2C conversion funnel. Use mobile app users as a high-intent audience for D2C campaigns.",
            ('Medium', 'High'): "Scale D2C success by improving mobile engagement. Create mobile-first content and app-exclusive offers.",
            ('Medium', 'Medium'): "Implement comprehensive optimization across both channels. Focus on improving user experience and conversion paths."
        }
        
        return strategies.get((mobile_engagement, d2c_conversion), "Develop integrated growth strategy focusing on customer lifecycle optimization.")
    
    def _get_budget_allocation_recommendation(self, mobile_free_percentage, d2c_roas):
        """Get budget allocation recommendation"""
        if mobile_free_percentage > 80 and d2c_roas > 3:
            return "Allocate 60% budget to D2C (high ROAS), 40% to mobile (freemium acquisition). Focus on converting mobile users to D2C customers."
        elif mobile_free_percentage > 80 and d2c_roas <= 3:
            return "Balance investment: 50% mobile user acquisition, 50% D2C optimization. Improve D2C funnel before scaling."
        elif mobile_free_percentage <= 80 and d2c_roas > 3:
            return "Premium mobile strategy: 40% mobile premium acquisition, 60% D2C scaling. Leverage premium mobile users for D2C growth."
        else:
            return "Conservative approach: Equal budget split with focus on optimization over acquisition until performance improves."
    
    def _create_integrated_report(self, mobile_report, d2c_report):
        """Create comprehensive integrated report"""
        integrated_report = {
            'executive_summary': self._create_executive_summary(mobile_report, d2c_report),
            'mobile_insights': mobile_report,
            'd2c_insights': d2c_report,
            'cross_platform_opportunities': self.cross_platform_insights,
            'integrated_recommendations': self._create_integrated_recommendations(),
            'next_steps': self._create_action_plan()
        }
        
        # Export integrated report
        import json
        with open('integrated_analytics_report.json', 'w') as f:
            json.dump(integrated_report, f, indent=2, default=str)
        
        print("📄 Integrated report exported to: integrated_analytics_report.json")
        
        return integrated_report
    
    def _create_executive_summary(self, mobile_report, d2c_report):
        """Create executive summary combining both analyses"""
        summary = {
            'report_date': datetime.now().isoformat(),
            'key_metrics': {},
            'top_opportunities': [],
            'critical_actions': []
        }
        
        # Key metrics
        if mobile_report:
            summary['key_metrics']['mobile_apps_analyzed'] = mobile_report.get('dataset_summary', {}).get('total_apps', 0)
            summary['key_metrics']['mobile_avg_rating'] = mobile_report.get('market_insights', {}).get('avg_rating_overall', 0)
        
        if d2c_report:
            summary['key_metrics']['d2c_segments_analyzed'] = d2c_report.get('funnel_analysis', {}).get('total_segments', 0)
            summary['key_metrics']['d2c_avg_roas'] = d2c_report.get('funnel_analysis', {}).get('average_metrics', {}).get('avg_roas', 0)
        
        # Top opportunities
        if mobile_report and mobile_report.get('top_opportunities'):
            summary['top_opportunities'].extend([
                f"Mobile: {opp['category']} category opportunity" 
                for opp in mobile_report['top_opportunities'][:2]
            ])
        
        if d2c_report and d2c_report.get('seo_analysis', {}).get('top_opportunities'):
            summary['top_opportunities'].extend([
                f"D2C SEO: {opp['category']} traffic potential" 
                for opp in d2c_report['seo_analysis']['top_opportunities'][:2]
            ])
        
        return summary
    
    def _create_integrated_recommendations(self):
        """Create integrated recommendations spanning both platforms"""
        return [
            {
                'priority': 'High',
                'area': 'Customer Journey Integration',
                'action': 'Implement unified customer tracking across mobile apps and D2C touchpoints',
                'expected_impact': 'Improved attribution accuracy and customer lifetime value optimization'
            },
            {
                'priority': 'High',
                'area': 'Cross-Platform Retargeting',
                'action': 'Create audiences from mobile app users for D2C campaigns and vice versa',
                'expected_impact': 'Higher conversion rates and reduced acquisition costs'
            },
            {
                'priority': 'Medium',
                'area': 'Content Strategy Alignment',
                'action': 'Align mobile app content with D2C SEO opportunities identified in analysis',
                'expected_impact': 'Improved organic discovery and brand consistency'
            },
            {
                'priority': 'Medium',
                'area': 'Technology Stack Integration',
                'action': 'Implement Customer Data Platform to unify mobile and D2C analytics',
                'expected_impact': 'Enhanced personalization and better decision-making capabilities'
            }
        ]
    
    def _create_action_plan(self):
        """Create 30-60-90 day action plan"""
        return {
            '30_days': [
                'Set up cross-platform tracking and attribution',
                'Create unified customer segments',
                'Launch cross-platform retargeting campaigns'
            ],
            '60_days': [
                'Implement customer data platform integration',
                'Optimize top-performing campaigns identified in analysis',
                'Launch content strategy alignment initiatives'
            ],
            '90_days': [
                'Complete funnel optimization based on insights',
                'Scale successful cross-platform strategies',
                'Establish ongoing performance monitoring and optimization processes'
            ]
        }

# Complete usage example
def run_full_analytics_suite(playstore_csv: str, d2c_excel: str, rapidapi_key: str = None):
    """
    Run the complete analytics suite with all phases
    """
    print("🎯 COMPLETE ANALYTICS SUITE")
    print("=" * 50)
    
    pipeline = IntegratedAnalyticsPipeline(rapidapi_key)
    
    results = pipeline.run_complete_pipeline(
        playstore_csv=playstore_csv,
        d2c_excel=d2c_excel,
        fetch_appstore=bool(rapidapi_key)
    )
    
    print("\n📋 SUMMARY OF DELIVERABLES:")
    print("- Mobile app market analysis and dashboard")
    print("- D2C funnel performance analysis")
    print("- SEO opportunity identification")
    print("- AI-generated creative content (headlines, descriptions, meta tags)")
    print("- Cross-platform insights and recommendations")
    print("- Integrated strategic roadmap")
    print("- JSON reports for all analyses")
    
    return results

# Final usage instructions
print("🚀 COMPLETE ANALYTICS PIPELINE READY!")
print("\nTo run the full analysis:")
print("results = run_full_analytics_suite('googleplaystore.csv', 'd2c_data.xlsx', 'YOUR_RAPIDAPI_KEY')")
print("\nTo run without App Store data:")
print("results = run_full_analytics_suite('googleplaystore.csv', 'd2c_data.xlsx')")
print("\nTo run only D2C analysis:")
print("analyzer, report = run_d2c_analysis('d2c_data.xlsx')")