# Twitter Virality Prediction System - Fogg Behavior Model

## Overview

This project implements a **Machine Learning system to predict tweet virality** using the **Fogg Behavior Model** as a theoretical framework for feature engineering.

### What is the Fogg Behavior Model?

The Fogg Behavior Model states: **Behavior = Motivation + Ability + Prompt**

This model helps us understand and predict user engagement by breaking down behavior into three dimensions:

1. **MOTIVATION** - Why people want to engage
   - Emotional triggers (sentiment, excitement, controversy)
   - Emojis and emotional language
   - Capital letters and punctuation (emphasis)

2. **ABILITY** - How easy it is to engage
   - Tweet length and readability
   - Presence of visual media (URLs)
   - Clear structure and formatting

3. **PROMPT** - What triggers engagement
   - Hashtags (#) - discoverable content
   - Mentions (@) - direct engagement
   - Questions (?) - prompts responses
   - Call-to-action keywords

4. **SOCIAL PROOF** - Existing engagement signals
   - Likes (immediate gratification)
   - Retweets (peer validation)
   - Engagement ratios

---

## Project Structure

```
fyp/
├── main.py                           # Main execution script
├── requirements.txt                  # Python dependencies
├── tweet_content-engagement_dataset.csv  # Input dataset
│
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py       # Fogg-aligned feature extraction
│   ├── model_training.py            # Model training & evaluation
│   └── visualization.py              # Analysis & visualization
│
├── models/
│   └── virality_model.pkl           # Trained model (generated)
│
├── results/
│   ├── virality_distribution.png    # Data analysis charts
│   ├── engagement_distribution.png
│   ├── feature_importance.png
│   ├── fogg_dimension_importance.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── model_metrics.png
│
└── notebooks/                        # Jupyter notebooks (optional)
```

---

## System Architecture

### Step 1: Data Loading
- Loads CSV dataset with tweet content and engagement metrics
- Analyzes virality distribution

### Step 2: Feature Engineering (Fogg Model)
- **MOTIVATION features**: Sentiment analysis, emotional words, emoji count, exclamation marks
- **ABILITY features**: Tweet length, word count, URL presence, readability metrics
- **PROMPT features**: Hashtag count, mention count, questions, call-to-action keywords
- **SOCIAL PROOF features**: Likes, retweets, engagement ratios

### Step 3: Data Preparation
- Splits data into training (80%) and test (20%) sets
- Normalizes features using StandardScaler
- Handles class imbalance via stratified sampling

### Step 4: Model Training
Trains and compares multiple ML models:
- **Random Forest** - Ensemble method with feature importance
- **Gradient Boosting** - Sequential ensemble learning
- **Logistic Regression** - Baseline linear model

### Step 5: Model Evaluation
Metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (Area Under the Curve)
- Confusion Matrix analysis

### Step 6: Feature Importance Analysis
- Identifies which features most strongly predict virality
- Analyzes importance by Fogg dimension
- Shows which behavioral factors matter most

### Step 7: Visualization & Report
Generates comprehensive visualizations:
- Virality distribution charts
- Engagement metrics analysis
- Feature importance rankings
- Model performance metrics
- Fogg dimension contribution analysis

---

## Dataset Structure

**tweet_content-engagement_dataset.csv**

| Column | Type | Description |
|--------|------|-------------|
| Tweet Content | string | The actual tweet text |
| Likes | integer | Number of likes received |
| Retweets | integer | Number of retweets received |
| Virality | string | Target: "Viral" or "Non-Viral" |
| Number of Hashtags | integer | Count of hashtags in tweet |

**Statistics:**
- 502 total tweets
- Features: Likes, Retweets, engagement metrics
- Target: Binary classification (Viral/Non-Viral)

---

## How to Run

### 1. Setup Environment

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data (done automatically in code)
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 2. Execute Pipeline

```bash
python main.py
```

This will:
1. Load and explore the dataset
2. Extract Fogg-aligned features
3. Train multiple models
4. Evaluate and select best model
5. Generate visualizations
6. Save trained model

### 3. View Results

Results are saved in the `results/` directory:
- **virality_distribution.png** - Tweet virality breakdown
- **engagement_distribution.png** - Likes/retweets patterns
- **feature_importance.png** - Top 15 important features
- **fogg_dimension_importance.png** - Contribution by Fogg dimension
- **confusion_matrix.png** - Prediction accuracy breakdown
- **roc_curve.png** - Model performance curve
- **model_metrics.png** - Accuracy, Precision, Recall, F1

---

## Key Features Extracted

### Motivation Features (9 features)
- `sentiment_compound`, `sentiment_positive`, `sentiment_negative`
- `emotion_score`, `emoji_count`, `has_emoji`
- `exclamation_count`, `question_count`, `capital_ratio`

### Ability Features (8 features)
- `tweet_length`, `word_count`, `avg_word_length`
- `has_url`, `url_count`, `has_quote`
- `is_short`, `is_long`

### Prompt Features (6 features)
- `hashtag_count`, `has_hashtag`
- `mention_count`, `has_mention`
- `cta_presence`, `is_question`

### Social Proof Features (6 features)
- `likes`, `retweets`, `total_engagement`
- `retweet_ratio`, `likes_log`, `retweets_log`

**Total: 29 engineered features**

---

## Model Performance Expectations

Based on the Fogg model framework:

- **Accuracy**: 75-85% (binary classification)
- **Precision**: 70-80% (correctly identified viral tweets)
- **Recall**: 65-75% (catch most viral tweets)
- **F1-Score**: 70-77% (balanced performance)
- **ROC-AUC**: 0.82-0.90 (strong discrimination)

---

## Fogg Model Insights

The system provides insights into which behavioral factors matter most for virality:

1. **Does MOTIVATION matter?** - Emotional content drives engagement
2. **Does ABILITY matter?** - Clarity and length affect shareability
3. **Does PROMPT matter?** - Hashtags and mentions improve discovery
4. **Which is most important?** - Feature importance analysis reveals the answer

---

## Usage Examples

### Making Predictions on New Tweets

```python
from src.feature_engineering import FoggFeatureExtractor
from src.model_training import ViraityPredictionModel
import pickle

# Load trained model
with open('models/virality_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Extract features from new tweet
extractor = FoggFeatureExtractor()
tweet = "Check out this amazing new feature! #Innovation #Tech"
features = extractor.extract_all_features(tweet)

# Predict virality
prediction = model.predict([features])
probability = model.predict_proba([features])
```

### Analyzing Feature Contributions

```python
feature_importance = model.feature_importances_
# Use visualization module to analyze by Fogg dimension
```

---

## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **matplotlib & seaborn**: Visualization
- **textblob & nltk**: Sentiment analysis
- **jupyter**: Interactive notebooks

---

## Future Enhancements

1. **Real-time Prediction API** - Serve predictions via REST API
2. **User Engagement Features** - Add account age, follower count
3. **Temporal Analysis** - Consider tweet timing and trends
4. **Deep Learning** - Use BERT/transformers for better NLP
5. **Explainability** - SHAP values for individual predictions
6. **Dashboard** - Interactive visualization platform

---

## Notes

- All features are normalized using StandardScaler for equal contribution
- Class imbalance is handled via stratified train/test split
- Random state (42) ensures reproducibility
- Models are trained on 80% of data, evaluated on 20% held-out test set

---

## Contact & References

**Fogg Behavior Model**: BJ Fogg's research on behavior change (http://www.behaviormodel.org/)

