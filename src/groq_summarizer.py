"""
Groq AI Summarizer for Virality Prediction Results
Uses Groq API for fast AI-powered insights with streaming support
"""

from groq import Groq


class GroqSummarizer:
    """
    Summarizes prediction results using Groq API
    Provides insights about tweet virality predictions with streaming support
    """
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key
            model: Model to use (llama-3.1-8b-instant, mixtral-8x7b-32768, gemma-7b-it)
        """
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def analyze_tweet_virality(self, tweet: str, prediction: str, probability: float) -> str:
        """
        Analyze tweet virality using Groq AI
        Simple method for streamlit app predictions
        
        Args:
            tweet: The tweet text
            prediction: "Viral" or "Non-Viral"
            probability: Probability score (0-1)
        
        Returns:
            String analysis from Groq
        """
        prompt = f"""You are a social media expert analyzing tweet virality using behavioral science principles.

ANALYZE THIS SPECIFIC TWEET:

**Tweet:** "{tweet}"

**Prediction:** {prediction}
**Confidence:** {probability:.1%}

Provide a concise 2-3 sentence analysis explaining:
- Why this tweet is predicted to be {prediction.lower()}
- Specific elements in the tweet that support this prediction
- One actionable suggestion to improve virality if needed

Be specific to THIS actual tweet, not generic advice."""

        message = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            stream=False,
        )
        
        return message.choices[0].message.content
    
    def summarize_prediction(self, tweet: str, prediction: str, probability: float, features: dict, stream: bool = False) -> str:
        """
        Generate AI summary of a tweet virality prediction
        
        Args:
            tweet: The tweet text
            prediction: "Viral" or "Non-Viral"
            probability: Probability score (0-1)
            features: Dictionary of extracted features
            stream: Whether to use streaming (not used in Streamlit version)
        
        Returns:
            String summary from Groq
        """
        # Build feature description dynamically from actual features
        feature_descriptions = []
        for key, value in sorted(features.items()):
            if isinstance(value, float):
                feature_descriptions.append(f"- {key.replace('_', ' ').title()}: {value:.2f}")
            else:
                feature_descriptions.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        features_text = "\n".join(feature_descriptions[:10])  # Limit to top 10 for clarity
        
        # Create specific guidance based on prediction
        prediction_guidance = f"""This tweet is predicted to be {prediction.lower()} with {probability:.1%} confidence.

IMPORTANT: Provide a SPECIFIC analysis of THIS PARTICULAR TWEET. Do not use generic templates.
- Reference specific words, tone, or hashtags from the tweet above
- Explain EXACTLY which features helped or hurt this prediction
- Suggest CONCRETE improvements if it's non-viral
- Highlight SPECIFIC strengths if it's viral
- DO NOT repeat the same analysis for different tweets"""
        
        prompt = f"""You are a social media analyst using the Fogg Behavior Model (Motivation × Ability × Prompt = Behavior).

ANALYZE THIS SPECIFIC TWEET:

**Tweet:** "{tweet}"

**Prediction:** {prediction}
**Confidence:** {probability:.1%}

**Extracted Features for This Tweet:**
{features_text}

{prediction_guidance}

Provide a 3-4 sentence expert analysis that is specific to this actual tweet, NOT generic advice."""

        message = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=350,
            stream=False,
        )
        
        return message.choices[0].message.content
    
    def summarize_model_results(self, metrics: dict, feature_importance_df, dimension_importance: dict, stream: bool = False) -> str:
        """
        Generate research summary of model performance
        
        Args:
            metrics: Model performance metrics (accuracy, precision, recall, f1, auc)
            feature_importance_df: DataFrame with top features
            dimension_importance: Dict with dimension importance scores
            stream: Whether to use streaming (not used in Streamlit version)
        
        Returns:
            String summary from Groq
        """
        top_features = feature_importance_df.head(5)['feature'].tolist()
        
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
            stream=False,
        )
        
        return message.choices[0].message.content
