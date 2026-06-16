"""
Standalone Model Training Script
Trains models once and saves them for inference
Run this script once: python train_models.py
"""

import sys
import os
import json
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_engineering import FoggFeatureExtractor, get_feature_groups
from model_training import DataPreprocessor, ViraityPredictionModel
import pandas as pd


def train_and_save_models():
    """
    Complete pipeline to train and save models for inference
    """
    
    # Configuration
    csv_path = 'tweet_content-engagement_dataset.csv'
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("TWITTER VIRALITY PREDICTION - MODEL TRAINING & SAVING")
    print("="*70)
    
    # ========== STEP 1: Load Data ==========
    print("\n[STEP 1] Loading Dataset...")
    preprocessor = DataPreprocessor(csv_path)
    df = preprocessor.load_data()
    
    # ========== STEP 2: Feature Engineering ==========
    print("\n[STEP 2] Feature Engineering...")
    extractor = FoggFeatureExtractor()
    df_engineered = extractor.process_dataset(df)
    print("✓ Features engineered successfully!")
    
    # Get feature groups and exclude social proof to prevent data leakage
    feature_groups = get_feature_groups()
    feature_columns = (
        feature_groups['motivation'] + 
        feature_groups['ability'] + 
        feature_groups['prompt']
    )
    
    print(f"✓ Total features: {len(feature_columns)}")
    
    # ========== STEP 3: Data Preparation ==========
    print("\n[STEP 3] Preparing data for training...")
    
    # Update preprocessor's dataframe with engineered features
    preprocessor.df = df_engineered
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_for_training(
        feature_columns,
        test_size=0.2,
        random_state=42
    )
    print("✓ Data split and scaled")
    
    # ========== STEP 4: Train and Save All Models ==========
    print("\n[STEP 4] Training models...")
    
    models_to_train = ['random_forest', 'gradient_boosting', 'logistic_regression']
    trained_models = {}
    results = {}
    
    for model_type in models_to_train:
        print(f"\n  Training {model_type}...")
        model = ViraityPredictionModel(model_type=model_type)
        model.build_model()
        model.train(X_train, y_train)
        model.predict(X_test)
        metrics = model.evaluate(y_test)
        
        # Save individual model
        model_path = os.path.join(models_dir, f'{model_type}_model.pkl')
        model.save_model(model_path)
        
        trained_models[model_type] = model
        results[model_type] = metrics
        
        print(f"  ✓ {model_type} saved!")
    
    # ========== STEP 5: Save Scaler and Preprocessor ==========
    print("\n[STEP 5] Saving scaler and preprocessor...")
    
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(preprocessor.scaler, f)
    print(f"  ✓ Scaler saved to {scaler_path}")
    
    # ========== STEP 6: Save Feature Columns Info ==========
    print("\n[STEP 6] Saving feature columns info...")
    
    feature_info = {
        'feature_columns': feature_columns,
        'motivation_features': feature_groups['motivation'],
        'ability_features': feature_groups['ability'],
        'prompt_features': feature_groups['prompt']
    }
    
    feature_info_path = os.path.join(models_dir, 'feature_info.json')
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"  ✓ Feature info saved to {feature_info_path}")
    
    # ========== STEP 7: Model Comparison & Report ==========
    print("\n[STEP 7] Model Performance Comparison")
    print("="*70)
    
    comparison_df = pd.DataFrame(results).T
    print("\n" + comparison_df.to_string())
    
    # Select best model
    best_model_type = comparison_df['f1'].idxmax()
    best_model = trained_models[best_model_type]
    
    print(f"\n✓ Best Model: {best_model_type.upper()} (F1: {results[best_model_type]['f1']:.4f})")
    
    # Save training report
    report_path = os.path.join(models_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write("TWITTER VIRALITY PREDICTION - TRAINING REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("Models Trained:\n")
        for model_type in models_to_train:
            f.write(f"  - {model_type}\n")
        f.write("\n" + "Model Performance:\n")
        f.write(comparison_df.to_string())
        f.write(f"\n\nBest Model: {best_model_type.upper()}\n")
        f.write(f"F1 Score: {results[best_model_type]['f1']:.4f}\n")
        f.write(f"Accuracy: {results[best_model_type]['accuracy']:.4f}\n")
        f.write(f"Precision: {results[best_model_type]['precision']:.4f}\n")
        f.write(f"Recall: {results[best_model_type]['recall']:.4f}\n")
    
    print(f"✓ Training report saved to {report_path}")
    
    print("\n" + "="*70)
    print("✓ All models trained and saved successfully!")
    print("✓ You can now run the Streamlit app for predictions")
    print("="*70 + "\n")


if __name__ == "__main__":
    train_and_save_models()
