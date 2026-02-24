"""
Main execution script for Twitter Virality Prediction System
Using Fogg Behavior Model framework
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_engineering import FoggFeatureExtractor, get_feature_groups
from model_training import DataPreprocessor, ViraityPredictionModel
from visualization import create_comprehensive_report
import pandas as pd


def main():
    """
    Main execution pipeline for virality prediction system
    """
    
    # Configuration
    csv_path = 'tweet_content-engagement_dataset.csv'
    model_output_path = 'models/virality_model.pkl'
    results_dir = 'results'
    
    print("\n" + "="*70)
    print("TWITTER VIRALITY PREDICTION SYSTEM - FOGG BEHAVIOR MODEL")
    print("="*70)
    
    # ========== STEP 1: Load Data ==========
    print("\n[STEP 1] Loading Dataset...")
    preprocessor = DataPreprocessor(csv_path)
    df = preprocessor.load_data()
    preprocessor.explore_data()
    
    # ========== STEP 2: Feature Engineering ==========
    print("\n" + "="*70)
    print("[STEP 2] Feature Engineering (Fogg Model)")
    print("="*70)
    
    extractor = FoggFeatureExtractor()
    print("\nExtracting features aligned with Fogg dimensions:")
    print("  - MOTIVATION: Emotional triggers (sentiment, emojis, excitement)")
    print("  - ABILITY: Content clarity (length, readability, media)")
    print("  - PROMPT: Call-to-action elements (hashtags, mentions, questions)")
    print("  - SOCIAL PROOF: Existing engagement signals")
    
    df_engineered = extractor.process_dataset(df)
    print(f"\n✓ Features engineered successfully!")
    print(f"  Total features created: {df_engineered.shape[1] - df.shape[1]}")
    
    # Get feature groups
    feature_groups = get_feature_groups()
    
    # IMPORTANT: Exclude social proof features to prevent data leakage
    # Predict virality based on CONTENT ONLY (Motivation + Ability + Prompt)
    feature_columns = (
        feature_groups['motivation'] + 
        feature_groups['ability'] + 
        feature_groups['prompt']
    )
    
    print("\nFeatures for Training (Content-Based):")
    print(f"  MOTIVATION: {len(feature_groups['motivation'])} features")
    print(f"  ABILITY: {len(feature_groups['ability'])} features")
    print(f"  PROMPT: {len(feature_groups['prompt'])} features")
    print(f"  TOTAL: {len(feature_columns)} features")
    print(f"\n⚠️  EXCLUDED Social Proof features (likes/retweets) to avoid data leakage")
    
    # ========== STEP 3: Data Preparation ==========
    print("\n" + "="*70)
    print("[STEP 3] Data Preparation")
    print("="*70)
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_for_training(
        feature_columns,
        test_size=0.2,
        random_state=42
    )
    
    print("\nClass distribution in training set:")
    print(f"  Non-Viral: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    print(f"  Viral: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
    
    # ========== STEP 4: Model Training ==========
    print("\n" + "="*70)
    print("[STEP 4] Model Training")
    print("="*70)
    
    # Train multiple models and compare
    models_to_train = ['random_forest', 'gradient_boosting', 'logistic_regression']
    trained_models = {}
    results = {}
    
    for model_type in models_to_train:
        print(f"\n--- Training {model_type.upper()} ---")
        model = ViraityPredictionModel(model_type=model_type)
        model.build_model()
        model.train(X_train, y_train)
        model.predict(X_test)
        metrics = model.evaluate(y_test)
        trained_models[model_type] = model
        results[model_type] = metrics
    
    # ========== STEP 5: Model Evaluation ==========
    print("\n" + "="*70)
    print("[STEP 5] Model Comparison")
    print("="*70)
    
    comparison_df = pd.DataFrame(results).T
    print("\n" + comparison_df.to_string())
    
    # Select best model
    best_model_type = comparison_df['f1'].idxmax()
    best_model = trained_models[best_model_type]
    
    print(f"\n✓ Best Model: {best_model_type.upper()}")
    best_model.print_evaluation_report(y_test)
    
    # ========== STEP 6: Feature Importance Analysis ==========
    print("\n" + "="*70)
    print("[STEP 6] Feature Importance Analysis")
    print("="*70)
    
    feature_importance_df = best_model.get_feature_importance(all_feature_columns)
    
    if feature_importance_df is not None:
        print("\nTop 15 Most Important Features:")
        print(feature_importance_df.head(15).to_string(index=False))
    
    # ========== STEP 7: Visualization & Report ==========
    print("\n" + "="*70)
    print("[STEP 7] Generating Visualizations & Report")
    print("="*70)
    
    create_comprehensive_report(
        df_engineered,
        best_model,
        X_test,
        y_test,
        feature_columns,
        feature_groups,
        output_dir=results_dir
    )
    
    # ========== STEP 8: Save Model ==========
    print("\n" + "="*70)
    print("[STEP 8] Saving Model")
    print("="*70)
    
    best_model.save_model(model_output_path)
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    print(f"✓ Dataset processed: {df.shape[0]} tweets")
    print(f"✓ Features engineered: {len(all_feature_columns)} features (Fogg-aligned)")
    print(f"✓ Best model: {best_model_type}")
    print(f"✓ Model accuracy: {best_model.metrics['accuracy']:.4f}")
    print(f"✓ Model ROC-AUC: {best_model.metrics['roc_auc']:.4f}")
    print(f"✓ Model saved: {model_output_path}")
    print(f"✓ Results saved: {results_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
