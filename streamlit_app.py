"""
Streamlit App for Twitter Virality Prediction System
Interactive interface for the Fogg Behavior Model implementation
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_engineering import FoggFeatureExtractor, get_feature_groups
from model_training import DataPreprocessor, ViraityPredictionModel
from visualization import AnalysisAndVisualization
from groq_summarizer import GroqSummarizer
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Twitter Virality Predictor",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_engineered' not in st.session_state:
    st.session_state.df_engineered = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'groq_summarizer' not in st.session_state:
    st.session_state.groq_summarizer = None
if 'groq_enabled' not in st.session_state:
    st.session_state.groq_enabled = False


# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.markdown("# 🐦 Virality Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Data Exploration", "⚙️ Feature Engineering", 
     "🤖 Model Training", "🔮 Make Predictions", "📈 Analysis"],
    help="Select a page to navigate"
)

st.sidebar.markdown("---")

# Groq API Key Setup
groq_api_key = os.getenv('GROQ_API_KEY')

# Auto-load Groq on startup
if groq_api_key and not st.session_state.groq_enabled:
    try:
        st.session_state.groq_summarizer = GroqSummarizer(groq_api_key)
        st.session_state.groq_enabled = True
    except Exception as e:
        st.session_state.groq_enabled = False

with st.sidebar.expander("⚡ Groq AI Setup", expanded=False):
    if st.session_state.groq_enabled:
        st.success("✅ Groq AI is ENABLED!")
        st.info("API key loaded from environment. AI summaries are active.")
        
        if st.button("🔄 Reload Groq"):
            st.rerun()
    else:
        st.markdown("**Enable AI Summarization using Groq**")
        st.info("Groq is fast, free, and open-source. Get your API key: https://console.groq.com/keys")
        
        api_key_input = st.text_input(
            "Groq API Key:",
            type="password",
            help="Get your free API key from https://console.groq.com/keys"
        )
        
        if api_key_input:
            with st.spinner("Validating API key..."):
                try:
                    st.session_state.groq_summarizer = GroqSummarizer(api_key_input)
                    st.session_state.groq_enabled = True
                    st.success("✅ Groq enabled! AI summaries available.")
                    st.info("💡 Tip: Add to .env file to auto-load on startup\nAdd: GROQ_API_KEY=your_key_here")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.info("""
            **To enable AI summaries:**
            
            Option 1 (Recommended):
            1. Get free key: https://console.groq.com/keys
            2. Create `.env` file in project root
            3. Add: `GROQ_API_KEY=your_key_here`
            4. Restart the app
            
            Option 2:
            1. Get free key above
            2. Paste it in the box above
            """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About This System
Uses **Fogg Behavior Model** to predict Twitter virality:
- **Motivation**: Emotional triggers
- **Ability**: Content clarity
- **Prompt**: Call-to-action elements
""")


# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.title("🐦 Twitter Virality Prediction System")
    st.markdown("### Using the Fogg Behavior Model Framework")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **What does this system do?**
        
        Predicts whether a tweet will go viral based on:
        - Content characteristics (length, emojis, sentiment)
        - Structural elements (hashtags, mentions, CTAs)
        - Engagement patterns (likes, retweets)
        """)
    
    with col2:
        st.success("""
        **Fogg Behavior Model**
        
        B = M + A + P
        
        - **M**otivation: Why engage?
        - **A**bility: How easy?
        - **P**rompt: What triggers?
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", "502 tweets", "Ready to load")
    with col2:
        st.metric("Features", "29 engineered", "Fogg-aligned")
    with col3:
        st.metric("Target", "Binary classification", "Viral/Non-Viral")
    
    st.markdown("---")
    st.markdown("""
    ### Quick Start
    1. **Data Exploration**: View dataset statistics and distributions
    2. **Feature Engineering**: Extract Fogg-aligned features
    3. **Model Training**: Train and compare ML models
    4. **Make Predictions**: Test on new tweets
    5. **Analysis**: Visualize results and feature importance
    
    Start by loading the dataset from the Data Exploration tab!
    """)


# ==================== DATA EXPLORATION ====================
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Load Dataset")
    
    with col2:
        if st.button("📥 Load Data", key="load_btn"):
            with st.spinner("Loading dataset..."):
                preprocessor = DataPreprocessor('tweet_content-engagement_dataset.csv')
                df = preprocessor.load_data()
                st.session_state.df = df
                st.session_state.preprocessor = preprocessor
                st.success("✓ Dataset loaded successfully!")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Basic statistics
        st.markdown("---")
        st.markdown("### Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tweets", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            viral_count = (df['Virality'] == 'Viral').sum()
            st.metric("Viral Tweets", viral_count)
        with col4:
            non_viral_count = (df['Virality'] == 'Non-Viral').sum()
            st.metric("Non-Viral Tweets", non_viral_count)
        
        st.markdown("---")
        st.markdown("### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Engagement Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Likes Statistics**")
            likes_stats = df['Likes'].describe()
            st.write(likes_stats)
        
        with col2:
            st.markdown("**Retweets Statistics**")
            retweets_stats = df['Retweets'].describe()
            st.write(retweets_stats)
        
        st.markdown("---")
        st.markdown("### Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Virality Distribution", "Engagement Distribution", "Hashtags Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Count plot
                fig, ax = plt.subplots(figsize=(8, 5))
                virality_counts = df['Virality'].value_counts()
                colors = ['#FF6B6B', '#4ECDC4']
                ax.bar(virality_counts.index, virality_counts.values, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_title('Tweet Count by Virality', fontsize=12, fontweight='bold')
                ax.set_ylabel('Count')
                ax.set_xlabel('Virality')
                st.pyplot(fig)
            
            with col2:
                # Percentage plot
                fig, ax = plt.subplots(figsize=(8, 5))
                virality_pct = df['Virality'].value_counts(normalize=True) * 100
                colors = ['#FF6B6B', '#4ECDC4']
                ax.pie(virality_pct.values, labels=virality_pct.index, autopct='%1.1f%%',
                       colors=colors, startangle=90)
                ax.set_title('Virality Distribution (%)', fontsize=12, fontweight='bold')
                st.pyplot(fig)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                viral = df[df['Virality'] == 'Viral']['Likes']
                non_viral = df[df['Virality'] == 'Non-Viral']['Likes']
                ax.hist([non_viral, viral], label=['Non-Viral', 'Viral'], 
                       bins=30, color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
                ax.set_xlabel('Number of Likes')
                ax.set_ylabel('Frequency')
                ax.set_title('Likes Distribution by Virality', fontweight='bold')
                ax.legend()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                viral_rt = df[df['Virality'] == 'Viral']['Retweets']
                non_viral_rt = df[df['Virality'] == 'Non-Viral']['Retweets']
                ax.hist([non_viral_rt, viral_rt], label=['Non-Viral', 'Viral'], 
                       bins=30, color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
                ax.set_xlabel('Number of Retweets')
                ax.set_ylabel('Frequency')
                ax.set_title('Retweets Distribution by Virality', fontweight='bold')
                ax.legend()
                st.pyplot(fig)
        
        with tab3:
            fig, ax = plt.subplots(figsize=(10, 5))
            hashtag_stats = df.groupby('Number of Hashtags')['Virality'].value_counts().unstack()
            hashtag_stats.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
            ax.set_title('Virality by Number of Hashtags', fontweight='bold')
            ax.set_xlabel('Number of Hashtags')
            ax.set_ylabel('Count')
            ax.legend(title='Virality')
            plt.xticks(rotation=0)
            st.pyplot(fig)


# ==================== FEATURE ENGINEERING ====================
elif page == "⚙️ Feature Engineering":
    st.title("⚙️ Feature Engineering (Fogg Model)")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please load data first from the Data Exploration page!")
    else:
        df = st.session_state.df
        
        st.markdown("""
        ### Fogg Behavior Model Features
        
        This system extracts features aligned with three behavioral dimensions:
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info("""
            **MOTIVATION**
            (9 features)
            - Sentiment
            - Emojis
            - Excitement
            """)
        
        with col2:
            st.info("""
            **ABILITY**
            (8 features)
            - Length
            - Readability
            - Media links
            """)
        
        with col3:
            st.info("""
            **PROMPT**
            (6 features)
            - Hashtags
            - Mentions
            - CTAs
            """)
        
        with col4:
            st.info("""
            **SOCIAL PROOF**
            (6 features)
            - Likes
            - Retweets
            - Ratios
            """)
        
        st.markdown("---")
        
        if st.button("🔧 Extract Features", key="extract_features"):
            with st.spinner("Extracting Fogg-aligned features..."):
                extractor = FoggFeatureExtractor()
                df_engineered = extractor.process_dataset(df)
                st.session_state.df_engineered = df_engineered
                
                # Get feature columns
                feature_groups = get_feature_groups()
                feature_columns = []
                for group_features in feature_groups.values():
                    feature_columns.extend(group_features)
                st.session_state.feature_columns = feature_columns
                
                st.success("✓ Features extracted successfully!")
        
        if st.session_state.df_engineered is not None:
            df_engineered = st.session_state.df_engineered
            feature_groups = get_feature_groups()
            
            st.markdown("---")
            st.markdown(f"### Feature Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Features", len(st.session_state.feature_columns), "29 features")
            with col2:
                st.metric("Original Columns", df.shape[1])
            with col3:
                st.metric("New Features", len(st.session_state.feature_columns))
            with col4:
                st.metric("Total Columns", df_engineered.shape[1])
            
            st.markdown("---")
            st.markdown("### Features by Fogg Dimension")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Motivation", "Ability", "Prompt", "Social Proof"])
            
            with tab1:
                st.markdown("**Motivation Features (9)**")
                features = feature_groups['motivation']
                for i, feature in enumerate(features, 1):
                    st.write(f"{i}. `{feature}`")
            
            with tab2:
                st.markdown("**Ability Features (8)**")
                features = feature_groups['ability']
                for i, feature in enumerate(features, 1):
                    st.write(f"{i}. `{feature}`")
            
            with tab3:
                st.markdown("**Prompt Features (6)**")
                features = feature_groups['prompt']
                for i, feature in enumerate(features, 1):
                    st.write(f"{i}. `{feature}`")
            
            with tab4:
                st.markdown("**Social Proof Features (6)**")
                features = feature_groups['social_proof']
                for i, feature in enumerate(features, 1):
                    st.write(f"{i}. `{feature}`")
            
            st.markdown("---")
            st.markdown("### Engineered Dataset Preview")
            st.dataframe(df_engineered.head(), use_container_width=True)


# ==================== MODEL TRAINING ====================
elif page == "🤖 Model Training":
    st.title("🤖 Model Training")
    
    if st.session_state.df_engineered is None:
        st.warning("⚠️ Please extract features first from the Feature Engineering page!")
    else:
        st.markdown("### Model Selection & Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("Select models to train:")
        
        with col2:
            train_rf = st.checkbox("Random Forest", value=True)
            train_gb = st.checkbox("Gradient Boosting", value=True)
            train_lr = st.checkbox("Logistic Regression", value=True)
        
        st.markdown("---")
        
        if st.button("🚀 Train Models", key="train_models"):
            df_engineered = st.session_state.df_engineered
            feature_groups = get_feature_groups()
            preprocessor = st.session_state.preprocessor
            
            # Update preprocessor's dataframe with engineered features
            preprocessor.df = df_engineered
            
            # IMPORTANT: Exclude social proof features (likes, retweets) to avoid data leakage
            # We want to predict virality based on CONTENT only, not actual engagement
            feature_columns_for_training = (
                feature_groups['motivation'] + 
                feature_groups['ability'] + 
                feature_groups['prompt']
            )
            st.session_state.feature_columns = feature_columns_for_training
            
            with st.spinner("Preparing data..."):
                X_train, X_test, y_train, y_test = preprocessor.prepare_for_training(
                    feature_columns_for_training,
                    test_size=0.2,
                    random_state=42
                )
                
                # Debug info
                st.info(f"""
                **Training Configuration:**
                - Features: {len(feature_columns_for_training)} (Motivation + Ability + Prompt)
                - Excluded: Social Proof features (likes/retweets) to prevent data leakage
                - Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}
                - Class balance: Viral={y_train.sum()} vs Non-Viral={(1-y_train).sum()} in train set
                """)
            
            models_to_train = []
            if train_rf:
                models_to_train.append('random_forest')
            if train_gb:
                models_to_train.append('gradient_boosting')
            if train_lr:
                models_to_train.append('logistic_regression')
            
            progress_bar = st.progress(0)
            trained_models = {}
            results = {}
            
            for idx, model_type in enumerate(models_to_train):
                with st.spinner(f"Training {model_type}..."):
                    model = ViraityPredictionModel(model_type=model_type)
                    model.build_model()
                    model.train(X_train, y_train)
                    model.predict(X_test)
                    metrics = model.evaluate(y_test)
                    
                    trained_models[model_type] = model
                    results[model_type] = metrics
                    
                    progress_bar.progress((idx + 1) / len(models_to_train))
            
            # Select best model
            comparison_df = pd.DataFrame(results).T
            best_model_type = comparison_df['f1'].idxmax()
            best_model = trained_models[best_model_type]
            
            st.session_state.model = best_model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.results = results
            
            st.success("✓ Training complete!")
        
        if st.session_state.model is not None and 'results' in st.session_state:
            st.markdown("---")
            st.markdown("### Model Comparison")
            
            comparison_df = pd.DataFrame(st.session_state.results).T
            st.dataframe(comparison_df, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Best Model Performance")
            
            best_model = st.session_state.model
            metrics = best_model.metrics
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}", delta="75-85% expected")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1']:.4f}")
            with col5:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            
            st.markdown("---")
            st.markdown("### Confusion Matrix")
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(st.session_state.y_test, best_model.predictions)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Non-Viral', 'Viral'],
                           yticklabels=['Non-Viral', 'Viral'],
                           cbar_kws={'label': 'Count'}, ax=ax)
                ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                st.pyplot(fig)
            
            with col2:
                st.markdown("""
                **Interpretation:**
                - **True Negatives**: Correctly identified non-viral tweets
                - **False Positives**: Non-viral tweets predicted as viral
                - **False Negatives**: Viral tweets predicted as non-viral
                - **True Positives**: Correctly identified viral tweets
                """)


# ==================== MAKE PREDICTIONS ====================
elif page == "🔮 Make Predictions":
    st.title("🔮 Make Predictions")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first from the Model Training page!")
    else:
        st.markdown("### Test Virality Predictions on New Tweets")
        
        # Example tweets
        example_tweets = [
            "Just discovered the most amazing coffee shop in town! ☕✨ #CoffeeLovers #DiscoveryOfTheDay",
            "Had a long day at work. Really exhausted.",
            "OMG THIS NEW FEATURE IS ABSOLUTELY INCREDIBLE!!! 🚀🔥 Click here to see it: http://example.com #GameChanger #MustSee @everyone",
            "Elections coming up. Make sure you're registered to vote!",
            "Cats are cool I guess.",
        ]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**Write or select a tweet to analyze:**")
        
        with col2:
            use_example = st.checkbox("Use example tweet")
        
        if use_example:
            selected_example = st.selectbox("Choose an example:", example_tweets)
            tweet_text = st.text_area("Tweet Content:", value=selected_example, height=100)
        else:
            tweet_text = st.text_area("Tweet Content:", height=100, placeholder="Enter a tweet here...")
        
        if st.button("🔮 Predict Virality", key="predict_btn"):
            if tweet_text.strip() == "":
                st.warning("⚠️ Please enter a tweet!")
            else:
                with st.spinner("Analyzing tweet..."):
                    # Extract features
                    extractor = FoggFeatureExtractor()
                    features = extractor.extract_all_features(tweet_text)
                    
                    # Get ONLY content-based feature columns (no social proof)
                    feature_groups = get_feature_groups()
                    feature_columns = (
                        feature_groups['motivation'] + 
                        feature_groups['ability'] + 
                        feature_groups['prompt']
                    )
                    
                    # Create feature vector with only content-based features
                    feature_dict = {col: features.get(col, 0) for col in feature_columns}
                    feature_df = pd.DataFrame([feature_dict])
                    
                    # Scale features
                    feature_scaled = st.session_state.preprocessor.scaler.transform(feature_df)
                    
                    # Make prediction
                    best_model = st.session_state.model
                    prediction = best_model.model.predict(feature_scaled)[0]
                    probability = best_model.model.predict_proba(feature_scaled)[0]
                    
                    st.markdown("---")
                    st.markdown("### Prediction Result")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if prediction == 1:
                            st.success("🚀 VIRAL")
                        else:
                            st.error("📊 NON-VIRAL")
                    
                    with col2:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Non-Viral Probability", f"{probability[0]:.2%}")
                        with col_b:
                            st.metric("Viral Probability", f"{probability[1]:.2%}")
                    
                    # Groq AI Summary
                    if st.session_state.groq_enabled:
                        st.markdown("---")
                        st.markdown("### ⚡ AI-Powered Insights (Groq)")
                        
                        with st.spinner("Generating AI insights..."):
                            try:
                                prediction_text = "Viral" if prediction == 1 else "Non-Viral"
                                summary = st.session_state.groq_summarizer.summarize_prediction(
                                    tweet=tweet_text,
                                    prediction=prediction_text,
                                    probability=probability[1],
                                    features=features,
                                    stream=False
                                )
                                st.markdown(summary)
                            except Exception as e:
                                st.error(f"Error generating summary: {str(e)}")
                    
                    st.markdown("---")
                    st.markdown("### Feature Analysis")
                    
                    # Feature breakdown by dimension
                    tab1, tab2, tab3, tab4 = st.tabs(["Motivation", "Ability", "Prompt", "Social Proof"])
                    
                    with tab1:
                        st.markdown("**Motivation Features** (Why engage?)")
                        for feat in feature_groups['motivation']:
                            st.write(f"• `{feat}`: {features.get(feat, 0)}")
                    
                    with tab2:
                        st.markdown("**Ability Features** (How easy?)")
                        for feat in feature_groups['ability']:
                            st.write(f"• `{feat}`: {features.get(feat, 0)}")
                    
                    with tab3:
                        st.markdown("**Prompt Features** (What triggers?)")
                        for feat in feature_groups['prompt']:
                            st.write(f"• `{feat}`: {features.get(feat, 0)}")
                    
                    with tab4:
                        st.markdown("**Social Proof Features** (Existing signals)")
                        for feat in feature_groups['social_proof']:
                            val = features.get(feat, 0)
                            st.write(f"• `{feat}`: {val}")


# ==================== ANALYSIS ====================
elif page == "📈 Analysis":
    st.title("📈 Analysis & Insights")
    
    if st.session_state.model is None or st.session_state.df_engineered is None:
        st.warning("⚠️ Please complete Model Training first!")
    else:
        st.markdown("### Feature Importance Analysis")
        
        best_model = st.session_state.model
        df_engineered = st.session_state.df_engineered
        feature_groups = get_feature_groups()
        
        # Use only content features (no social proof)
        feature_columns = (
            feature_groups['motivation'] + 
            feature_groups['ability'] + 
            feature_groups['prompt']
        )
        y_test = st.session_state.y_test
        
        st.info("📌 **Note:** Social Proof features (likes/retweets) are excluded to avoid data leakage. Analysis is based on content features only.")
        
        st.markdown("---")
        
        # Feature importance
        feature_importance_df = best_model.get_feature_importance(feature_columns)
        
        if feature_importance_df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Top 15 Important Features")
                top_15 = feature_importance_df.head(15)
                st.dataframe(top_15, use_container_width=True)
                
                # Plot top 15
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(data=top_15, y='feature', x='importance', palette='viridis', ax=ax)
                ax.set_title('Top 15 Most Important Features', fontsize=12, fontweight='bold')
                ax.set_xlabel('Importance')
                st.pyplot(fig)
            
            with col2:
                st.markdown("### Fogg Dimension Importance")
                
                dimension_importance = {}
                for dimension, features in feature_groups.items():
                    dimension_features = feature_importance_df[
                        feature_importance_df['feature'].isin(features)
                    ]
                    dimension_importance[dimension] = dimension_features['importance'].sum()
                
                # Display metrics
                dim_df = pd.DataFrame(list(dimension_importance.items()), 
                                     columns=['Dimension', 'Total Importance'])
                dim_df = dim_df.sort_values('Total Importance', ascending=False)
                
                for _, row in dim_df.iterrows():
                    st.metric(row['Dimension'].upper(), f"{row['Total Importance']:.4f}")
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
                bars = ax.bar(dim_df['Dimension'], dim_df['Total Importance'], 
                             color=colors, edgecolor='black', linewidth=1.5)
                ax.set_title('Feature Importance by Fogg Dimension', fontsize=12, fontweight='bold')
                ax.set_ylabel('Total Importance')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### ROC Curve")
        
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, best_model.probabilities)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fpr, tpr, color='#4ECDC4', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Most Important Fogg Dimension:**
            
            The Fogg dimension with the highest total feature importance 
            reveals which behavioral factor matters most for Twitter virality.
            """)
        
        with col2:
            top_dimension = dim_df.iloc[0]
            st.success(f"""
            **{top_dimension['Dimension'].upper()}** is the strongest predictor!
            
            Importance: {top_dimension['Total Importance']:.4f}
            """)
        
        # Groq AI Summary of Results
        if st.session_state.groq_enabled:
            st.markdown("---")
            st.markdown("### ⚡ AI-Powered Research Summary (Groq)")
            
            with st.spinner("Generating research summary..."):
                try:
                    summary = st.session_state.groq_summarizer.summarize_model_results(
                        metrics=best_model.metrics,
                        feature_importance_df=feature_importance_df,
                        dimension_importance=dict(zip(dim_df['Dimension'], dim_df['Total Importance'])),
                        stream=False
                    )
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p style="color: gray;">Twitter Virality Prediction System | Powered by Fogg Behavior Model</p>
</div>
""", unsafe_allow_html=True)
