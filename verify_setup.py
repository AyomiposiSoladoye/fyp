"""
Setup and Verification Script
Checks if models are trained and helps initialize the system
"""

import os
import sys
import json

def check_models_exist():
    """Check if trained models exist"""
    models_dir = 'models'
    required_files = [
        'random_forest_model.pkl',
        'gradient_boosting_model.pkl',
        'logistic_regression_model.pkl',
        'scaler.pkl',
        'feature_info.json',
        'training_report.txt'
    ]
    
    print("\n" + "="*70)
    print("CHECKING FOR TRAINED MODELS")
    print("="*70 + "\n")
    
    missing = []
    for filename in required_files:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} - MISSING")
            missing.append(filename)
    
    if missing:
        print("\n" + "⚠️  MODELS NOT FOUND!")
        print("\nTo train the models, run:")
        print("  python train_models.py")
        return False
    else:
        print("\n✓ All models are ready!")
        return True


def verify_feature_extraction():
    """Verify feature extraction works"""
    print("\n" + "="*70)
    print("VERIFYING FEATURE EXTRACTION")
    print("="*70 + "\n")
    
    try:
        sys.path.insert(0, 'src')
        from feature_engineering import FoggFeatureExtractor
        
        extractor = FoggFeatureExtractor()
        print("✓ Feature extractor loaded successfully")
        
        # Try extracting features from a sample tweet
        import pandas as pd
        sample_df = pd.DataFrame({
            'Tweet Content': ['This is an amazing tweet! #exciting #trending 🚀']
        })
        
        result = extractor.process_dataset(sample_df)
        print(f"✓ Feature extraction works ({len(result.columns)} features created)")
        
        return True
    except Exception as e:
        print(f"✗ Feature extraction failed: {str(e)}")
        return False


def verify_model_inference():
    """Verify model inference works"""
    print("\n" + "="*70)
    print("VERIFYING MODEL INFERENCE")
    print("="*70 + "\n")
    
    try:
        sys.path.insert(0, 'src')
        from model_inference import get_predictor
        
        predictor = get_predictor(model_type='random_forest')
        print("✓ Model loaded successfully")
        
        # Try a sample prediction
        test_tweet = "Check out this amazing new feature! 🎉 #tech #innovation"
        result = predictor.predict_single_tweet(test_tweet)
        
        if result['error']:
            print(f"✗ Prediction failed: {result['error']}")
            return False
        else:
            print(f"✓ Prediction successful")
            print(f"  Tweet: '{test_tweet}'")
            print(f"  Virality: {result['virality']}")
            print(f"  Confidence: {result['confidence']}")
            return True
    
    except Exception as e:
        print(f"✗ Model inference failed: {str(e)}")
        return False


def main():
    """Run all verification checks"""
    
    print("\n" + "="*70)
    print("TWITTER VIRALITY PREDICTION SYSTEM - SETUP VERIFICATION")
    print("="*70)
    
    # Check models
    models_ok = check_models_exist()
    if not models_ok:
        return False
    
    # Verify feature extraction
    features_ok = verify_feature_extraction()
    if not features_ok:
        print("\n⚠️  Feature extraction verification failed")
        return False
    
    # Verify model inference
    inference_ok = verify_model_inference()
    if not inference_ok:
        print("\n⚠️  Model inference verification failed")
        return False
    
    # All checks passed
    print("\n" + "="*70)
    print("✓ ALL CHECKS PASSED!")
    print("="*70)
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run streamlit_app.py")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
