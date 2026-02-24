"""
Groq AI Summarizer for Virality Prediction Results
Uses Groq API for fast AI-powered insights
"""

from groq import Groq


class GroqSummarizer:
    """
    Summarizes prediction results using Groq API
    Provides insights about tweet virality predictions
    """
    
    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768"):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key
            model: Model to use (mixtral-8x7b-32768, llama-2-70b-chat, etc.)
        """
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def summarize_prediction(self, tweet: str, prediction: str, probability: float, features: dict) -> str:
        """
        Generate AI summary of a tweet virality prediction
        
        Args:
            tweet: The tweet text
            prediction: "Viral" or "Non-Viral"
            probability: Probability score (0-1)
            features: Dictionary of extracted features
        
        Returns:
            String summary from Groq
        """
        prompt = f"""You are a social media analyst using the Fogg Behavior Model (Motivation × Ability × Prompt = Behavior).

Analyze this tweet prediction:

**Tweet:** "{tweet}"

**Prediction:** {prediction}
**Confidence:** {probability:.1%}

Based on the Fogg model dimensions:
- Motivation: {features.get('sentiment_score', 0):.2f} (emotional appeal)
- Ability: {features.get('avg_word_length', 0):.2f} (clarity/readability)
- Prompt: {features.get('hashtag_count', 0)} hashtags, {features.get('mention_count', 0)} mentions (CTAs)

Provide a 2-3 sentence expert analysis of why this tweet would or would not go viral, focusing on behavioral triggers."""

        message = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        
        return message.choices[0].message.content
    
    def summarize_model_results(self, metrics: dict, feature_importance_df, dimension_importance: dict) -> str:
        """
        Generate research summary of model performance
        
        Args:
            metrics: Model performance metrics (accuracy, precision, recall, f1, auc)
            feature_importance_df: DataFrame with top features
            dimension_importance: Dict with dimension importance scores
        
        Returns:
            String summary from Groq
        """
        top_features = feature_importance_df.head(5)['Feature'].tolist()
        
        prompt = f"""You are a research analyst examining a Twitter virality prediction model using the Fogg Behavior Model.

**Model Performance:**
- Accuracy: {metrics.get('accuracy', 0):.1%}
- Precision: {metrics.get('precision', 0):.1%}
- Recall: {metrics.get('recall', 0):.1%}
- F1-Score: {metrics.get('f1', 0):.1%}
- ROC-AUC: {metrics.get('auc', 0):.1%}

**Top Predictive Features:**
{', '.join(top_features)}

**Fogg Dimension Importance:**
- Motivation: {dimension_importance.get('Motivation', 0):.1%}
- Ability: {dimension_importance.get('Ability', 0):.1%}
- Prompt: {dimension_importance.get('Prompt', 0):.1%}

Provide a 3-4 sentence research insight about what drives Twitter virality according to this model and its alignment with behavioral science."""

        message = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
        )
        
        return message.choices[0].message.content
