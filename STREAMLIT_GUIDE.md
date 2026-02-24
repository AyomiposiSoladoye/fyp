# Quick Start Guide - Streamlit App

## Installation & Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## App Navigation

The Streamlit app has **6 main pages** accessible from the left sidebar:

### 1. 🏠 **Home**
- Overview of the system
- Fogg Behavior Model explanation
- Quick start instructions
- System statistics

### 2. 📊 **Data Exploration**
- Load the dataset
- View dataset statistics
- Explore engagement metrics
- Visualize virality distribution
- Analyze hashtag patterns

### 3. ⚙️ **Feature Engineering**
- Extract Fogg-aligned features
- View all 29 engineered features
- Organized by Fogg dimension
- See feature descriptions

### 4. 🤖 **Model Training**
- Select models to train (Random Forest, Gradient Boosting, Logistic Regression)
- Train all models
- Compare model performance
- View confusion matrix
- See detailed metrics

### 5. 🔮 **Make Predictions**
- Test virality on new tweets
- Use example tweets or write your own
- Get instant predictions
- View feature breakdown
- See probability scores

### 6. 📈 **Analysis**
- Feature importance ranking
- Fogg dimension contribution analysis
- ROC curve visualization
- Key insights and findings

---

## Workflow

### Option A: Quick Test (5 minutes)
1. Go to **Data Exploration** → Click "Load Data"
2. Go to **Feature Engineering** → Click "Extract Features"
3. Go to **Model Training** → Click "Train Models"
4. Go to **Make Predictions** → Write a tweet and predict

### Option B: Full Analysis (15 minutes)
1. **Data Exploration** - Understand the dataset
2. **Feature Engineering** - See how features are built
3. **Model Training** - Train and compare models
4. **Make Predictions** - Test on sample tweets
5. **Analysis** - Understand feature importance

### Option C: Command Line Only (no Streamlit)
```bash
python main.py
```
This runs the full pipeline and saves results to the `results/` folder

---

## Understanding the Output

### In Data Exploration
- **Virality Distribution**: How many tweets are viral vs non-viral
- **Engagement Distribution**: Typical likes/retweets patterns
- **Hashtag Analysis**: Effect of hashtag count on virality

### In Model Training
- **Accuracy**: Overall correctness of predictions
- **Precision**: Accuracy of viral predictions
- **Recall**: What percentage of viral tweets we catch
- **F1-Score**: Balance between precision and recall
- **Confusion Matrix**: True positives/negatives breakdown

### In Make Predictions
- **Prediction Result**: Viral 🚀 or Non-Viral 📊
- **Probability Scores**: Confidence levels
- **Feature Analysis**: Breakdown by Fogg dimension

### In Analysis
- **Feature Importance**: Which features matter most
- **Fogg Dimension Importance**: Which behavioral factor matters most
- **ROC Curve**: Model discrimination ability

---

## Example Tweets to Test

### High Virality Potential
- "Just discovered the most amazing feature ever!!! 🚀✨ #Innovation #GameChanger"
- "OMG this is INCREDIBLE!! Check this out: http://example.com #MustSee"
- "Can't believe how awesome this is! 😍💯 #Trending"

### Low Virality Potential
- "Had a day. Nothing special happened."
- "I guess crypto is a thing."
- "another meeting today"

---

## Tips & Tricks

1. **Session Persistence**: Your progress is saved as you move through pages
2. **Example Tweets**: Use the predefined examples to see how different styles perform
3. **Real-time Testing**: Make predictions instantly on any tweet
4. **Feature Inspection**: Click on features to understand what they measure

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Run `pip install streamlit`

### Problem: "FileNotFoundError: tweet_content-engagement_dataset.csv"
**Solution:** Make sure the CSV file is in the same directory as `streamlit_app.py`

### Problem: "NLTK data missing"
**Solution:** This is auto-downloaded, but you can manually run:
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Problem: "TextBlob errors"
**Solution:** Download corpora:
```bash
python -m textblob.download_corpora
```

---

## Performance Notes

- **First run**: May take 30-60 seconds (NLTK data downloading)
- **Feature extraction**: ~5-10 seconds for 502 tweets
- **Model training**: ~10-20 seconds (Random Forest is slowest)
- **Predictions**: ~1-2 seconds per tweet

---

## Features Extracted

### MOTIVATION (9 features)
- Sentiment analysis (compound, positive, negative)
- Emotion score
- Emoji count and presence
- Exclamation marks and questions
- Capital letter ratio

### ABILITY (8 features)
- Tweet length and word count
- Average word length
- URL presence and count
- Quote marks
- Short/long indicators

### PROMPT (6 features)
- Hashtag count and presence
- Mention count and presence
- Call-to-action presence
- Question presence

### SOCIAL PROOF (6 features)
- Likes and retweets (raw and log)
- Total engagement
- Retweet ratio

---

## Next Steps

1. **Test different tweet styles** - See what makes tweets go viral
2. **Analyze results** - Check which Fogg dimension matters most
3. **Export findings** - Save visualizations from the results folder
4. **Improve features** - Add new features aligned with Fogg model
5. **Deploy model** - Create a web API for predictions

