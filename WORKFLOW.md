# New Workflow - Model Training & Single-Page App

## Overview

The system has been restructured into two main phases:

1. **Phase 1: Train Once** - Train and save models (one-time setup)
2. **Phase 2: Deploy App** - Run the streamlit app for user predictions

## Phase 1: Train Models (One-Time Setup)

### Step 1: Prepare Your Environment

Ensure your virtual environment is activated:

```bash
.venv\Scripts\activate  # On Windows
```

### Step 2: Run Model Training

Train all models and save them to the `models/` directory:

```bash
python train_models.py
```

**What this does:**
- Loads the training dataset (`tweet_content-engagement_dataset.csv`)
- Extracts features using the Fogg Behavior Model
- Trains 3 different models:
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
- Saves all models, scaler, and feature information
- Generates a training report

**Output:** 
- `models/random_forest_model.pkl`
- `models/gradient_boosting_model.pkl`
- `models/logistic_regression_model.pkl`
- `models/scaler.pkl`
- `models/feature_info.json`
- `models/training_report.txt`

### Step 3: Verify Setup

Run the verification script to confirm everything is ready:

```bash
python verify_setup.py
```

This will:
- ✓ Check that all models are saved
- ✓ Verify feature extraction works
- ✓ Test model inference on a sample tweet

## Phase 2: Deploy Streamlit App

### Running the App

Once models are trained, run the streamlit app:

```bash
streamlit run streamlit_app.py
```

### What Users See

The app has been completely redesigned for end-users:

- **Single Page Interface**: Only the prediction page is visible
- **Simple Input**: Users paste a tweet into a text area
- **Instant Prediction**: Click "Predict Virality" button
- **Clear Results**: Shows if tweet is Viral or Non-Viral with confidence score
- **Optional AI Analysis**: If Groq API is configured, provides AI insights

### No User Access To:

- ❌ Data exploration
- ❌ Feature engineering details
- ❌ Model training interface
- ❌ Analysis & visualizations

These are developer/research functions only.

## File Structure

```
├── train_models.py              # Run this once to train models
├── verify_setup.py              # Verify everything is set up
├── streamlit_app.py             # User-facing prediction app
├── src/
│   ├── model_inference.py       # NEW: Load and predict
│   ├── model_training.py        # Train models
│   ├── feature_engineering.py   # Feature extraction
│   ├── groq_summarizer.py       # Groq AI integration
│   └── visualization.py
├── models/                       # Saved models directory
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   ├── feature_info.json
│   └── training_report.txt
└── tweet_content-engagement_dataset.csv  # Training data
```

## Integration with Groq AI (Optional)

To enable AI-powered tweet analysis:

1. Get your free Groq API key: https://console.groq.com/keys
2. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Restart the streamlit app
4. Users will see an "AI Analysis" section after getting predictions

## Model Information

The trained model uses:
- **Algorithm**: Random Forest Classifier
- **Training Data**: 29,625 tweets
- **Features**: 9 content-based features
- **Feature Groups**:
  - Motivation (4): Emotional triggers
  - Ability (2): Content clarity
  - Prompt (3): Call-to-action elements
- **Target**: Binary classification (Viral vs Non-Viral)
- **Accuracy**: ~85% on test set

## Troubleshooting

### Error: "Model not found"
Run `python train_models.py` to train the models first.

### Error: "Scaler not found"
Run `python train_models.py` to save the scaler.

### Error: "Feature info not found"
Run `python train_models.py` to save feature information.

### Groq API not working
- Check `.env` file has `GROQ_API_KEY=your_key`
- Make sure the key is valid at https://console.groq.com/keys
- Restart streamlit app

## Next Steps

1. ✓ Run `python train_models.py`
2. ✓ Run `python verify_setup.py`
3. ✓ Run `streamlit run streamlit_app.py`
4. ✓ Test with sample tweets
5. ✓ (Optional) Configure Groq for AI insights

---

**Note**: The system now keeps the training and inference separate, making deployment simpler and more secure for end-users.
