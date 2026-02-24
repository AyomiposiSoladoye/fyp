"""
Feature Engineering Module for Fogg Behavior Model
Maps tweet characteristics to Fogg dimensions: Motivation, Ability, and Prompt
"""

import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


class FoggFeatureExtractor:
    """
    Extracts features aligned with Fogg Behavior Model:
    - MOTIVATION: Emotional triggers and sentiment
    - ABILITY: Content clarity and structure
    - PROMPT: Call-to-action elements
    """
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.emotion_words = {
            'love': 1, 'amazing': 1, 'awesome': 1, 'great': 1, 'excellent': 1,
            'hate': -1, 'terrible': -1, 'awful': -1, 'bad': -1, 'horrible': -1
        }
        self.emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"
                                       u"\U0001F300-\U0001F5FF"
                                       u"\U0001F680-\U0001F6FF"
                                       u"\U0001F1E0-\U0001F1FF"
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       u"\U0001f926-\U0001f937"
                                       u"\U00010000-\U0010ffff"
                                       u"\u2640-\u2642"
                                       u"\u2600-\u2B55"
                                       u"\u200d"
                                       u"\u23cf"
                                       u"\u23e9"
                                       u"\u231a"
                                       u"\ufe0f"
                                       u"\u3030"
                                       "]+", re.UNICODE)
    
    def extract_motivation_features(self, tweet):
        """
        MOTIVATION: Why people want to engage
        - Sentiment intensity
        - Emotional language
        - Emoji usage
        - Controversial/trending topics
        """
        features = {}
        
        # Sentiment analysis
        sentiment_scores = self.sia.polarity_scores(tweet)
        features['sentiment_compound'] = sentiment_scores['compound']
        features['sentiment_positive'] = sentiment_scores['pos']
        features['sentiment_negative'] = sentiment_scores['neg']
        
        # Emotion words count
        tweet_lower = tweet.lower()
        emotion_score = sum([
            self.emotion_words[word] 
            for word in self.emotion_words 
            if word in tweet_lower
        ])
        features['emotion_score'] = emotion_score
        
        # Emoji presence
        emojis = self.emoji_pattern.findall(tweet)
        features['emoji_count'] = len(emojis)
        features['has_emoji'] = 1 if emojis else 0
        
        # Exclamation marks (excitement indicator)
        features['exclamation_count'] = tweet.count('!')
        features['question_count'] = tweet.count('?')
        
        # Capital letters (emphasis)
        capital_letters = sum(1 for c in tweet if c.isupper())
        features['capital_ratio'] = capital_letters / max(len(tweet), 1)
        
        return features
    
    def extract_ability_features(self, tweet):
        """
        ABILITY: How easy it is to engage
        - Tweet length/readability
        - Presence of multimedia references
        - Clarity and structure
        """
        features = {}
        
        # Length features
        features['tweet_length'] = len(tweet)
        features['word_count'] = len(tweet.split())
        features['avg_word_length'] = np.mean([len(word) for word in tweet.split()]) if tweet.split() else 0
        
        # URL/Media presence (easier to engage with visual content)
        features['has_url'] = 1 if ('http' in tweet or 'www' in tweet) else 0
        features['url_count'] = len(re.findall(r'http\S+|www\S+', tweet))
        
        # Quote/reference patterns
        features['has_quote'] = 1 if '"' in tweet else 0
        
        # Readability (shorter, simpler tweets are easier to engage)
        features['is_short'] = 1 if len(tweet) < 100 else 0
        features['is_long'] = 1 if len(tweet) > 200 else 0
        
        return features
    
    def extract_prompt_features(self, tweet):
        """
        PROMPT: What triggers engagement
        - Hashtags
        - Mentions
        - Direct calls-to-action
        """
        features = {}
        
        # Hashtags
        hashtags = re.findall(r'#\w+', tweet)
        features['hashtag_count'] = len(hashtags)
        features['has_hashtag'] = 1 if hashtags else 0
        
        # Mentions
        mentions = re.findall(r'@\w+', tweet)
        features['mention_count'] = len(mentions)
        features['has_mention'] = 1 if mentions else 0
        
        # Call-to-action indicators
        cta_keywords = ['check', 'click', 'read', 'watch', 'vote', 'share', 'follow', 'subscribe', 'visit']
        cta_count = sum(1 for keyword in cta_keywords if keyword in tweet.lower())
        features['cta_presence'] = cta_count
        
        # Question (prompts response)
        features['is_question'] = 1 if '?' in tweet else 0
        
        return features
    
    def extract_social_proof_features(self, df_row):
        """
        SOCIAL PROOF: Existing engagement signals
        - Likes and retweets as indicators
        """
        features = {}
        
        if 'Likes' in df_row:
            features['likes'] = df_row['Likes']
            features['likes_log'] = np.log1p(df_row['Likes'])
        
        if 'Retweets' in df_row:
            features['retweets'] = df_row['Retweets']
            features['retweets_log'] = np.log1p(df_row['Retweets'])
        
        # Engagement ratio
        if 'Likes' in df_row and 'Retweets' in df_row:
            total_engagement = df_row['Likes'] + df_row['Retweets']
            features['total_engagement'] = total_engagement
            features['retweet_ratio'] = df_row['Retweets'] / max(df_row['Likes'], 1)
        
        return features
    
    def extract_all_features(self, tweet, row=None):
        """
        Extract all Fogg-aligned features from a tweet
        """
        all_features = {}
        
        # Motivation features
        all_features.update(self.extract_motivation_features(tweet))
        
        # Ability features
        all_features.update(self.extract_ability_features(tweet))
        
        # Prompt features
        all_features.update(self.extract_prompt_features(tweet))
        
        # Social proof features (if row data available)
        if row is not None:
            all_features.update(self.extract_social_proof_features(row))
        
        return all_features
    
    def process_dataset(self, df):
        """
        Process entire dataset and extract features for all tweets
        Returns DataFrame with all engineered features
        """
        features_list = []
        
        for idx, row in df.iterrows():
            tweet = row['Tweet Content']
            features = self.extract_all_features(tweet, row)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Combine with original data
        result_df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
        
        return result_df


def get_feature_groups():
    """
    Returns feature groups organized by Fogg dimension
    """
    feature_groups = {
        'motivation': [
            'sentiment_compound', 'sentiment_positive', 'sentiment_negative',
            'emotion_score', 'emoji_count', 'has_emoji', 'exclamation_count',
            'question_count', 'capital_ratio'
        ],
        'ability': [
            'tweet_length', 'word_count', 'avg_word_length', 'has_url',
            'url_count', 'has_quote', 'is_short', 'is_long'
        ],
        'prompt': [
            'hashtag_count', 'has_hashtag', 'mention_count', 'has_mention',
            'cta_presence', 'is_question'
        ],
        'social_proof': [
            'likes', 'retweets', 'total_engagement', 'retweet_ratio',
            'likes_log', 'retweets_log'
        ]
    }
    return feature_groups
