"""
Gemini AI Summarization Module
Uses Google Gemini API to summarize virality predictions and analysis results
"""

from google import genai
from typing import Dict, List
import os


class GeminiSummarizer:
    """
    Uses Gemini API to provide AI-generated insights and summaries
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini with API key
        
        Args:
            api_key: Google Generative AI API key
                    If None, will look for GOOGLE_API_KEY environment variable
        """
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            raise ValueError(
                "Gemini API key not provided. "
                "Please pass api_key parameter or set GOOGLE_API_KEY environment variable. "
                "Get your key from: https://aistudio.google.com/app/apikey"
            )
        
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-1.5-flash"
    
    def summarize_prediction(self, tweet: str, prediction: str, probability: float, 
                            features: Dict) -> str:
        """
        Summarize a virality prediction with insights
        
        Args:
            tweet: The tweet text
            prediction: "Viral" or "Non-Viral"
            probability: Probability score (0-1)
            features: Dictionary of feature values
        
        Returns:
            AI-generated summary and insights
        """
        
        motivation_score = (
            (features.get('sentiment_compound', 0) + 1) / 2 * 0.3 +  # -1 to 1 → 0 to 1
            features.get('emoji_count', 0) / 3 * 0.2 +
            features.get('exclamation_count', 0) / 5 * 0.2 +
            (1 if features.get('has_emoji', 0) else 0) * 0.3
        ) / 3
        
        ability_score = (
            (1 if 50 < features.get('tweet_length', 0) < 200 else 0) * 0.4 +
            (1 if features.get('has_url', 0) else 0) * 0.3 +
            features.get('avg_word_length', 5) / 10
        ) / 3
        
        prompt_score = (
            (1 if features.get('hashtag_count', 0) > 0 else 0) * 0.4 +
            (1 if features.get('is_question', 0) else 0) * 0.3 +
            features.get('cta_presence', 0) / 3
        ) / 3
        
        prompt = f"""
Analyze this Twitter virality prediction and provide insights:

TWEET: "{tweet}"

PREDICTION RESULT:
- Prediction: {prediction}
- Confidence: {probability:.1%}

FOGG BEHAVIOR MODEL ANALYSIS:
- Motivation Score (emotional triggers): {motivation_score:.2f}/1.0
- Ability Score (content clarity): {ability_score:.2f}/1.0
- Prompt Score (call-to-action): {prompt_score:.2f}/1.0

KEY FEATURES:
- Sentiment: {features.get('sentiment_compound', 0):.2f}
- Emojis: {features.get('emoji_count', 0)}
- Hashtags: {features.get('hashtag_count', 0)}
- Has URL: {bool(features.get('has_url', 0))}
- Is Question: {bool(features.get('is_question', 0))}
- Exclamation Marks: {features.get('exclamation_count', 0)}
- Tweet Length: {features.get('tweet_length', 0)} chars

Please provide:
1. A brief explanation of why this tweet is predicted as {prediction}
2. Which Fogg dimension (Motivation/Ability/Prompt) is strongest
3. Specific suggestions to improve virality (2-3 actionable recommendations)
4. An overall virality potential assessment (Low/Medium/High)

Keep response concise and actionable (3-4 sentences).
"""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text
    
    def summarize_model_results(self, metrics: Dict, feature_importance_df, 
                               dimension_importance: Dict) -> str:
        """
        Summarize overall model performance and feature importance
        
        Args:
            metrics: Dictionary of model metrics (accuracy, precision, etc.)
            feature_importance_df: DataFrame of feature importances
            dimension_importance: Dictionary of Fogg dimension importances
        
        Returns:
            AI-generated summary of model results
        """
        
        top_features = feature_importance_df.head(5)['feature'].tolist()
        top_dimension = max(dimension_importance, key=dimension_importance.get)
        
        prompt = f"""
Summarize these machine learning results for a Twitter virality prediction system using the Fogg Behavior Model:

MODEL PERFORMANCE:
- Accuracy: {metrics.get('accuracy', 0):.1%}
- Precision: {metrics.get('precision', 0):.1%}
- Recall: {metrics.get('recall', 0):.1%}
- F1-Score: {metrics.get('f1', 0):.1%}
- ROC-AUC: {metrics.get('roc_auc', 0):.1%}

TOP 5 IMPORTANT FEATURES:
{', '.join(top_features)}

FOGG DIMENSION IMPORTANCE:
- Motivation: {dimension_importance.get('motivation', 0):.4f}
- Ability: {dimension_importance.get('ability', 0):.4f}
- Prompt: {dimension_importance.get('prompt', 0):.4f}

STRONGEST DIMENSION: {top_dimension.upper()}

Please provide:
1. Overall model quality assessment (Good/Fair/Needs Improvement)
2. Key insight about which Fogg dimension matters most for Twitter virality
3. Main limitation or consideration for using this model
4. One interesting finding about tweet characteristics that drive virality

Keep to 3-4 sentences, academic tone suitable for a thesis.
"""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text
    
    def generate_research_insights(self, prediction_results: List[Dict]) -> str:
        """
        Generate research insights from multiple predictions
        
        Args:
            prediction_results: List of prediction dictionaries
        
        Returns:
            AI-generated research insights
        """
        
        summary_stats = {
            'total_predictions': len(prediction_results),
            'viral_count': sum(1 for p in prediction_results if p.get('prediction') == 'Viral'),
            'avg_confidence': sum(p.get('probability', 0) for p in prediction_results) / max(len(prediction_results), 1)
        }
        
        prompt = f"""
Analyze these tweet virality predictions and provide research insights:

SUMMARY STATISTICS:
- Total Tweets Analyzed: {summary_stats['total_predictions']}
- Predicted Viral: {summary_stats['viral_count']}
- Predicted Non-Viral: {summary_stats['total_predictions'] - summary_stats['viral_count']}
- Average Prediction Confidence: {summary_stats['avg_confidence']:.1%}

Based on the Fogg Behavior Model (Motivation + Ability + Prompt), please:
1. Identify patterns in what makes tweets go viral
2. Discuss the relationship between the three Fogg dimensions
3. Provide insights about social media engagement from a behavioral science perspective
4. Suggest how creators could use these insights to improve content

Keep response academic and suitable for thesis research (5-7 sentences).
"""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text


def validate_api_key(api_key: str) -> bool:
    """
    Validate Gemini API key by attempting a simple request
    
    Args:
        api_key: Google Generative AI API key
    
    Returns:
        True if key is valid, False otherwise
    """
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents="Say 'valid' in one word."
        )
        return bool(response.text)
    except Exception as e:
        print(f"API key validation failed: {e}")
        return False
