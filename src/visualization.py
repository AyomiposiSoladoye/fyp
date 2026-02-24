"""
Analysis and Visualization Module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os


class AnalysisAndVisualization:
    """
    Create visualizations and analysis for the virality prediction system
    """
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
    
    def plot_virality_distribution(self, df):
        """Plot distribution of Viral vs Non-Viral tweets"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Count plot
        virality_counts = df['Virality'].value_counts()
        axes[0].bar(virality_counts.index, virality_counts.values, color=['#FF6B6B', '#4ECDC4'])
        axes[0].set_title('Tweet Virality Distribution', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count')
        
        # Percentage plot
        virality_pct = df['Virality'].value_counts(normalize=True) * 100
        axes[1].pie(virality_pct.values, labels=virality_pct.index, autopct='%1.1f%%',
                    colors=['#FF6B6B', '#4ECDC4'], startangle=90)
        axes[1].set_title('Virality Percentage', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'virality_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: virality_distribution.png")
        plt.close()
    
    def plot_engagement_metrics(self, df):
        """Plot likes and retweets by virality"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Likes distribution
        viral = df[df['Virality'] == 'Viral']['Likes']
        non_viral = df[df['Virality'] == 'Non-Viral']['Likes']
        
        axes[0].hist([non_viral, viral], label=['Non-Viral', 'Viral'], 
                     bins=30, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
        axes[0].set_xlabel('Number of Likes')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Likes Distribution by Virality', fontweight='bold')
        axes[0].legend()
        
        # Retweets distribution
        viral_rt = df[df['Virality'] == 'Viral']['Retweets']
        non_viral_rt = df[df['Virality'] == 'Non-Viral']['Retweets']
        
        axes[1].hist([non_viral_rt, viral_rt], label=['Non-Viral', 'Viral'], 
                     bins=30, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
        axes[1].set_xlabel('Number of Retweets')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Retweets Distribution by Virality', fontweight='bold')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'engagement_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: engagement_distribution.png")
        plt.close()
    
    def plot_feature_importance(self, feature_importance_df, top_n=15):
        """Plot top N important features"""
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: feature_importance.png")
        plt.close()
    
    def plot_fogg_dimension_importance(self, feature_importance_df, feature_groups):
        """Analyze feature importance by Fogg dimension"""
        dimension_importance = {}
        
        for dimension, features in feature_groups.items():
            dimension_features = feature_importance_df[
                feature_importance_df['feature'].isin(features)
            ]
            dimension_importance[dimension] = dimension_features['importance'].sum()
        
        plt.figure(figsize=(10, 6))
        dimensions = list(dimension_importance.keys())
        importances = list(dimension_importance.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        bars = plt.bar(dimensions, importances, color=colors, edgecolor='black', linewidth=1.5)
        plt.title('Feature Importance by Fogg Dimension', fontsize=14, fontweight='bold')
        plt.ylabel('Total Importance')
        plt.xlabel('Fogg Dimension')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fogg_dimension_importance.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: fogg_dimension_importance.png")
        plt.close()
        
        return dimension_importance
    
    def plot_confusion_matrix(self, y_test, predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Viral', 'Viral'],
                   yticklabels=['Non-Viral', 'Viral'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: confusion_matrix.png")
        plt.close()
    
    def plot_roc_curve(self, y_test, probabilities):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#4ECDC4', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: roc_curve.png")
        plt.close()
    
    def plot_model_metrics(self, metrics):
        """Plot model performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for idx, metric in enumerate(metric_names):
            value = metrics[metric]
            axes[idx].barh(['Score'], [value], color=colors[idx], height=0.5)
            axes[idx].set_xlim([0, 1])
            axes[idx].set_title(f'{metric.capitalize()}: {value:.4f}', fontweight='bold')
            axes[idx].set_xlabel('Score')
            axes[idx].text(value/2, 0, f'{value:.4f}', ha='center', va='center', 
                          fontweight='bold', color='white', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_metrics.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: model_metrics.png")
        plt.close()


def create_comprehensive_report(df, model, X_test, y_test, feature_columns, feature_groups, output_dir='results'):
    """
    Create a comprehensive analysis report
    """
    analyzer = AnalysisAndVisualization(output_dir)
    
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    # Basic data visualizations
    analyzer.plot_virality_distribution(df)
    analyzer.plot_engagement_metrics(df)
    
    # Model performance visualizations
    analyzer.plot_confusion_matrix(y_test, model.predictions)
    analyzer.plot_roc_curve(y_test, model.probabilities)
    analyzer.plot_model_metrics(model.metrics)
    
    # Feature importance visualizations
    feature_importance_df = model.get_feature_importance(feature_columns)
    if feature_importance_df is not None:
        analyzer.plot_feature_importance(feature_importance_df)
        dimension_importance = analyzer.plot_fogg_dimension_importance(feature_importance_df, feature_groups)
        
        print("\n=== Feature Importance by Fogg Dimension ===")
        for dimension, importance in dimension_importance.items():
            print(f"{dimension.capitalize()}: {importance:.4f}")
    
    print(f"\nAll visualizations saved to '{output_dir}' directory")
