"""
Data Processing and Model Training Module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import pickle
import os


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing
    """
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        """Load CSV file"""
        self.df = pd.read_csv(self.csv_path)
        print(f"Dataset loaded: {self.df.shape[0]} tweets, {self.df.shape[1]} features")
        return self.df
    
    def explore_data(self):
        """Display basic data exploration"""
        print("\n=== Dataset Overview ===")
        print(self.df.head())
        print("\n=== Data Info ===")
        print(self.df.info())
        print("\n=== Virality Distribution ===")
        print(self.df['Virality'].value_counts())
        print("\nVirality percentages:")
        print(self.df['Virality'].value_counts(normalize=True) * 100)
    
    def prepare_for_training(self, feature_columns, test_size=0.2, random_state=42):
        """
        Prepare data for training by splitting into train/test sets
        """
        # Separate features and target
        X = self.df[feature_columns].copy()
        
        # Encode target variable
        y = (self.df['Virality'] == 'Viral').astype(int)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=feature_columns,
            index=self.X_train.index
        )
        
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=feature_columns,
            index=self.X_test.index
        )
        
        print(f"\nTraining set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Features: {len(feature_columns)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test


class ViraityPredictionModel:
    """
    Machine learning model for virality prediction
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.predictions = None
        self.probabilities = None
        self.metrics = {}
    
    def build_model(self):
        """Initialize model based on type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\nTraining {self.model_type} model...")
        self.model.fit(X_train, y_train)
        print("Model training completed!")
    
    def predict(self, X_test):
        """Make predictions"""
        self.predictions = self.model.predict(X_test)
        self.probabilities = self.model.predict_proba(X_test)[:, 1]
        return self.predictions, self.probabilities
    
    def evaluate(self, y_test):
        """Evaluate model performance"""
        accuracy = accuracy_score(y_test, self.predictions)
        precision = precision_score(y_test, self.predictions)
        recall = recall_score(y_test, self.predictions)
        f1 = f1_score(y_test, self.predictions)
        roc_auc = roc_auc_score(y_test, self.probabilities)
        
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        return self.metrics
    
    def print_evaluation_report(self, y_test):
        """Print detailed evaluation metrics"""
        print("\n" + "="*50)
        print("MODEL EVALUATION REPORT")
        print("="*50)
        
        print(f"\nAccuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1-Score:  {self.metrics['f1']:.4f}")
        print(f"ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, self.predictions, 
                                   target_names=['Non-Viral', 'Viral']))
        
        print("\n=== Confusion Matrix ===")
        cm = confusion_matrix(y_test, self.predictions)
        print(f"True Negatives:  {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives:  {cm[1,1]}")
    
    def get_feature_importance(self, feature_columns):
        """Get feature importance from trained model"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df
        else:
            return None
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")
