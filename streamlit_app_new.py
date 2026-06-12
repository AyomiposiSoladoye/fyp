"""
Streamlit App for Twitter Virality Prediction
Single-page app for users to enter tweets and get virality predictions
"""

import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_inference import get_predictor
from groq_summarizer import GroqSummarizer

# Page config
st.set_page_config(
    page_title="Twitter Virality Prediction",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding: 2rem 1rem;
    }
    .stTextArea > label {
        font-size: 18px;
        font-weight: bold;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .viral-badge {
        background: #FF6B6B;
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
    }
    .non-viral-badge {
        background: #4ECDC4;
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'groq_enabled' not in st.session_state:
    st.session_state.groq_enabled = False
if 'groq_summarizer' not in st.session_state:
    st.session_state.groq_summarizer = None

# ==================== MAIN CONTENT ====================
st.title("🐦 Twitter Virality Prediction")
st.markdown("### Predict if your tweet will go viral")
st.markdown("---")

# Initialize predictor
with st.spinner("Loading model..."):
    try:
        if st.session_state.predictor is None:
            st.session_state.predictor = get_predictor(model_type='random_forest')
        predictor = st.session_state.predictor
    except FileNotFoundError as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Please run `python train_models.py` first to train and save the models.")
        st.stop()

# Tweet input section
st.markdown("### Enter your tweet:")
tweet_text = st.text_area(
    "Tweet content",
    placeholder="Type or paste your tweet here...",
    height=150,
    label_visibility="collapsed"
)

# Predict button
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_button = st.button(
        "🔮 Predict Virality",
        use_container_width=True,
        type="primary"
    )

# Make prediction
if predict_button:
    if not tweet_text.strip():
        st.warning("Please enter a tweet to make a prediction.")
    else:
        with st.spinner("Analyzing tweet..."):
            result = predictor.predict_single_tweet(tweet_text)
            
            if result['error']:
                st.error(f"❌ Error: {result['error']}")
            else:
                # Display prediction
                st.markdown("---")
                st.markdown("### Prediction Result:")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    virality_status = result['virality']
                    probability = result['probability']
                    confidence = result['confidence']
                    
                    if virality_status == 'Viral':
                        st.markdown(f"""
                        <div class="viral-badge">
                            🚀 VIRAL<br/>
                            {confidence}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="non-viral-badge">
                            📊 NON-VIRAL<br/>
                            {confidence}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.metric(
                        "Probability Score",
                        f"{probability:.4f}",
                        delta=f"{probability*100:.1f}% viral likelihood"
                    )
                
                # Additional info
                st.markdown("---")
                
                # Groq AI Summary (if enabled)
                groq_api_key = os.getenv('GROQ_API_KEY')
                
                if groq_api_key and not st.session_state.groq_enabled:
                    try:
                        st.session_state.groq_summarizer = GroqSummarizer(groq_api_key)
                        st.session_state.groq_enabled = True
                    except:
                        st.session_state.groq_enabled = False
                
                if st.session_state.groq_enabled and st.session_state.groq_summarizer:
                    with st.expander("💡 AI Analysis (Powered by Groq)"):
                        try:
                            analysis = st.session_state.groq_summarizer.analyze_tweet_virality(
                                tweet_text,
                                virality_status,
                                probability
                            )
                            st.markdown(analysis)
                        except Exception as e:
                            st.warning(f"Could not generate AI analysis: {str(e)}")

# ==================== SIDEBAR INFO ====================
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This prediction system uses machine learning to analyze your tweet and predict the likelihood of it going viral.

**How it works:**
- Analyzes content structure (hashtags, mentions, URLs)
- Evaluates sentiment and emotional language
- Assesses engagement triggers
- Provides a virality score

**Note:** Predictions are based on content analysis only and don't include social proof factors like existing engagement.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Info:**
- Algorithm: Random Forest Classifier
- Training Data: 29,625 tweets
- Features: 9 content-based features
- Accuracy: ~85%
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px; margin-top: 2rem;'>
    🐦 Twitter Virality Prediction System<br/>
    © 2024 | Powered by Machine Learning
</div>
""", unsafe_allow_html=True)
