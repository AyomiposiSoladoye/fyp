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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling - Dark theme with blue accents
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: #0f1419 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    }
    
    [data-testid="stMainBlockContainer"] {
        padding: 0;
        max-width: 800px;
        margin: 0 auto;
        background: #0f1419;
    }
    
    .main {
        padding: 0;
        background: #0f1419;
    }
    
    /* Title */
    .main-title {
        text-align: center;
        padding: 40px 20px 20px;
        border-bottom: 1px solid #2f3f4f;
        margin-bottom: 20px;
    }
    
    .main-title h1 {
        font-size: 2.2rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    /* Input Container */
    .input-container {
        padding: 0 20px 20px;
        max-width: 100%;
    }
    
    .stTextArea {
        width: 100%;
    }
    
    .stTextArea textarea {
        background: #192734 !important;
        color: #ffffff !important;
        border: 1px solid #38444d !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        padding: 12px !important;
        font-family: inherit !important;
        resize: vertical !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #8899a6 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #1d9bf0 !important;
        box-shadow: 0 0 0 3px rgba(29, 155, 240, 0.2) !important;
        outline: none !important;
    }
    
    /* Button */
    .stButton > button {
        width: 100%;
        background: #1d9bf0 !important;
        color: white !important;
        border: none !important;
        border-radius: 24px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        margin-top: 8px !important;
    }
    
    .stButton > button:hover {
        background: #1a8cd8 !important;
    }
    
    .stButton > button:active {
        opacity: 0.8 !important;
    }
    
    /* Results */
    .result-section {
        padding: 20px;
        text-align: center;
    }
    
    .result-status {
        font-size: 1.8rem;
        font-weight: 900;
        margin-bottom: 8px;
        color: #ffffff;
        animation: fadeInScale 0.4s ease;
    }
    
    .result-label {
        font-size: 16px;
        color: #8899a6;
        margin-bottom: 12px;
        font-weight: 500;
    }
    
    .result-score {
        font-size: 2.8rem;
        font-weight: 900;
        color: #1d9bf0;
        margin: 8px 0;
    }
    
    .result-score-label {
        font-size: 13px;
        color: #8899a6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    .result-bar {
        background: #2f3f4f;
        height: 4px;
        border-radius: 2px;
        margin: 12px 0;
        overflow: hidden;
    }
    
    .result-bar-fill {
        height: 100%;
        background: #1d9bf0;
        animation: fillBar 0.6s ease;
    }
    
    @keyframes fillBar {
        from { width: 0; }
        to { width: 100%; }
    }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* AI Analysis */
    .ai-box {
        background: #192734;
        border-left: 3px solid #1d9bf0;
        border-radius: 12px;
        padding: 16px;
        margin-top: 16px;
        text-align: left;
        font-size: 14px;
        line-height: 1.6;
        color: #ffffff;
    }
    
    .ai-title {
        font-weight: 700;
        margin-bottom: 8px;
        color: #1d9bf0;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    /* Hide streamlit elements */
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    .stDeployButton {
        visibility: hidden;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: transparent !important;
        border: 1px solid #efefef !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        margin-top: 24px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f7f9fa !important;
    }
    
    .streamlit-expanderHeader p {
        color: #1d9bf0 !important;
        font-weight: 600 !important;
        margin: 0 !important;
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

# Title
st.markdown("""
<div class="main-title">
    <h1>Twitter Virality Prediction System</h1>
</div>
""", unsafe_allow_html=True)

# Initialize predictor
with st.spinner("Loading model..."):
    try:
        if st.session_state.predictor is None:
            st.session_state.predictor = get_predictor(model_type='random_forest')
        predictor = st.session_state.predictor
    except FileNotFoundError as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

# Input Section
st.markdown('<div class="input-container">', unsafe_allow_html=True)

tweet_text = st.text_area(
    "label",
    placeholder="Enter your tweet",
    height=60,
    label_visibility="collapsed"
)

predict_button = st.button("Predict", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Results
if predict_button:
    if not tweet_text.strip():
        st.error("Enter a tweet to analyze")
    else:
        with st.spinner(""):
            result = predictor.predict_single_tweet(tweet_text)
            
            if result['error']:
                st.error(f"Error: {result['error']}")
            else:
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                
                virality_status = result['virality']
                probability = result['probability']
                
                # Status emoji with text
                if virality_status == 'Viral':
                    st.markdown('<div class="result-status">🚀 Your tweet is likely to go viral!</div>', unsafe_allow_html=True)
                    status_text = "Viral"
                    bar_width = probability * 100
                else:
                    st.markdown('<div class="result-status">📊 Your tweet is unlikely to go viral</div>', unsafe_allow_html=True)
                    status_text = "Non-Viral"
                    bar_width = (1 - probability) * 100
                
                # Score - Clear percentage display
                viral_prob_percent = probability * 100
                st.markdown(f'<div class="result-score">{viral_prob_percent:.0f}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-score-label">Viral Probability</div>', unsafe_allow_html=True)
                
                # Progress bar
                st.markdown(f"""
                <div class="result-bar">
                    <div class="result-bar-fill" style="width: {probability*100:.0f}%;"></div>
                </div>
                """, unsafe_allow_html=True)
                
                # AI Analysis
                groq_api_key = os.getenv('GROQ_API_KEY')
                
                if groq_api_key and not st.session_state.groq_enabled:
                    try:
                        st.session_state.groq_summarizer = GroqSummarizer(groq_api_key)
                        st.session_state.groq_enabled = True
                    except:
                        st.session_state.groq_enabled = False
                
                if st.session_state.groq_enabled and st.session_state.groq_summarizer:
                    try:
                        with st.spinner(""):
                            analysis = st.session_state.groq_summarizer.analyze_tweet_virality(
                                tweet_text,
                                virality_status,
                                probability
                            )
                        st.markdown(f"""
                        <div class="ai-box">
                            <div class="ai-title">⚡ AI-Powered Analysis</div>
                            {analysis}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning("⚠️ AI analysis unavailable - please check your Groq API key")
                
                st.markdown('</div>', unsafe_allow_html=True)
