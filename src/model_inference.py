"""
Model Inference Module
Loads trained models and makes predictions on new tweets
"""

import os
import json
import pickle
import pandas as pd
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_engineering import FoggFeatureExtractor


class ViraityPredictor:
    """
    Loads pre-trained models and makes predictions on new tweets
    """
    
    def __init__(self, models_dir='models', model_type='random_forest'):
        """
        Initialize predictor with pre-trained model
        
        Args:
            models_dir: Directory containing saved models
            model_type: Which model to use ('random_forest', 'gradient_boosting', 'logistic_regression')
        """
        self.models_dir = models_dir
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.extractor = FoggFeatureExtractor()
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, scaler, and feature info from disk"""
        
        # Load model
        model_path = os.path.join(self.models_dir, f'{self.model_type}_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run train_models.py first.")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Loaded {self.model_type} model")
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run train_models.py first.")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✓ Loaded scaler")
        
        # Load feature info
        feature_info_path = os.path.join(self.models_dir, 'feature_info.json')
        if not os.path.exists(feature_info_path):
            raise FileNotFoundError(f"Feature info not found at {feature_info_path}. Run train_models.py first.")
        
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)
        self.feature_columns = feature_info['feature_columns']
        print(f"✓ Loaded feature columns ({len(self.feature_columns)} features)")
    
    def predict_single_tweet(self, tweet_text):
        """
        Make a prediction for a single tweet
        
        Args:
            tweet_text: The tweet content to predict
            
        Returns:
            dict: Prediction result with probability and virality status
        """
        
        # Create a minimal DataFrame with the tweet
        df_tweet = pd.DataFrame({'Tweet Content': [tweet_text]})
        
        # Extract features
        try:
            df_features = self._extract_features(df_tweet)
        except Exception as e:
            return {
                'error': f"Feature extraction failed: {str(e)}",
                'prediction': None,
                'probability': None,
                'virality': None
            }
        
        # Make prediction
        try:
            X = df_features[self.feature_columns].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0][1]
            
            result = {
                'error': None,
                'prediction': int(prediction),
                'probability': float(probability),
                'virality': 'Viral' if prediction == 1 else 'Non-Viral',
                'confidence': f"{probability * 100:.1f}%"
            }
            
            return result
        
        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'prediction': None,
                'probability': None,
                'virality': None
            }
    
    def _extract_features(self, df):
        """
        Extract features from tweet dataframe
        Uses the same process as training
        """
        # Rename column if needed
        if 'Tweet Content' in df.columns:
            df = df.copy()
        else:
            # Handle alternative column names
            if 'content' in df.columns:
                df = df.copy()
                df['Tweet Content'] = df['content']
            elif 'tweet' in df.columns:
                df = df.copy()
                df['Tweet Content'] = df['tweet']
        
        # Use the extractor to process the tweet
        # The extractor processes a single tweet
        df_processed = self.extractor.process_dataset(df)
        
        return df_processed
    
    def predict_batch(self, tweets_list):
        """
        Make predictions for multiple tweets
        
        Args:
            tweets_list: List of tweet texts
            
        Returns:
            List of prediction results
        """
        results = []
        for tweet in tweets_list:
            result = self.predict_single_tweet(tweet)
            results.append(result)
        return results


# Singleton instance for streamlit
_predictor_instance = None


def get_predictor(model_type='random_forest'):
    """
    Get or create a predictor instance (singleton for streamlit)
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = ViraityPredictor(model_type=model_type)
    
    return _predictor_instance
