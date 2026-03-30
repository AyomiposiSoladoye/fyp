import sys
sys.path.insert(0, './src')

from feature_engineering import FoggFeatureExtractor
from model_training import DataPreprocessor, ViraityPredictionModel
import pandas as pd
import pickle

example_tweets = [
    "Just discovered the most amazing coffee shop in town! ☕✨ #CoffeeLovers #DiscoveryOfTheDay",
    "Had a long day at work. Really exhausted.",
    "OMG THIS NEW FEATURE IS ABSOLUTELY INCREDIBLE!!! 🚀🔥 Click here to see it: http://example.com #GameChanger #MustSee @everyone",
    "Elections coming up. Make sure you're registered to vote!",
    "Cats are cool I guess.",
]

extractor = FoggFeatureExtractor()

print("=" * 80)
print("FEATURE EXTRACTION TEST")
print("=" * 80)

for i, tweet in enumerate(example_tweets, 1):
    print(f"\n{i}. Tweet: {tweet[:60]}...")
    features = extractor.extract_all_features(tweet)
    
    # Check key features
    print(f"   Sentiment: {features.get('sentiment_compound', 0):.2f}")
    print(f"   Emoji count: {features.get('emoji_count', 0)}")
    print(f"   Hashtag count: {features.get('hashtag_count', 0)}")
    print(f"   Exclamation count: {features.get('exclamation_count', 0)}")
    print(f"   Mention count: {features.get('mention_count', 0)}")
    print(f"   Capital ratio: {features.get('capital_ratio', 0):.2f}")
    print(f"   Emotion score: {features.get('emotion_score', 0)}")

# Now try loading the model and making predictions
print("\n\n" + "=" * 80)
print("PREDICTION TEST")
print("=" * 80)

try:
    # Load preprocessor
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load model
    with open('models/best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    
    print("\nModel loaded successfully!")
    
    # Get feature columns (content-based only)
    from feature_engineering import get_feature_groups
    feature_groups = get_feature_groups()
    feature_columns = (
        feature_groups['motivation'] + 
        feature_groups['ability'] + 
        feature_groups['prompt']
    )
    
    for i, tweet in enumerate(example_tweets, 1):
        features = extractor.extract_all_features(tweet)
        feature_dict = {col: features.get(col, 0) for col in feature_columns}
        feature_df = pd.DataFrame([feature_dict])
        
        # Scale features
        feature_scaled = preprocessor.scaler.transform(feature_df)
        
        # Make prediction
        prediction = best_model.model.predict(feature_scaled)[0]
        probability = best_model.model.predict_proba(feature_scaled)[0]
        
        label = "VIRAL" if prediction == 1 else "NON-VIRAL"
        print(f"\n{i}. Prediction: {label}")
        print(f"   Probability: Non-Viral={probability[0]:.2%}, Viral={probability[1]:.2%}")
        
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure to train the model first!")
