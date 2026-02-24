# Groq AI Integration Guide

## What is Groq?
Groq provides **fast, free inference** for open-source AI models. Perfect for real-time applications!

**Key Benefits:**
- ✅ Free API (generous rate limits)
- ✅ Fast inference (great for live predictions)
- ✅ Multiple models: Mixtral, Llama 2, and more
- ✅ No credit card required for free tier

## Setup Instructions

### Step 1: Get Your API Key
1. Go to https://console.groq.com/keys
2. Sign up with your email (free account)
3. Click "Create API Key"
4. Copy the key (starts with `gsk_`)

### Step 2: Configure in Your Project

**Option A: Environment File (Recommended)**
```bash
# Create a .env file in your project root
# Copy the template:
cp .env.example .env

# Edit .env and add your key:
GROQ_API_KEY=gsk_your_key_here
```

**Option B: Use in Streamlit UI**
1. Run the Streamlit app
2. Click the "⚡ Groq AI Setup" sidebar expander
3. Paste your API key
4. Click "Validate"

### Step 3: Test It Works
The app will auto-detect your Groq API key and enable AI summaries on:
- **Make Predictions page**: Analyze why a tweet will/won't go viral
- **Analysis page**: Get research insights about your model

## Models Available
By default, Groq Summarizer uses `mixtral-8x7b-32768`. You can change this:

```python
# In streamlit_app.py or your code:
summarizer = GroqSummarizer(
    api_key="your_key",
    model="llama-2-70b-chat"  # Alternative models
)
```

**Available models:**
- `mixtral-8x7b-32768` (Fast, balanced)
- `llama-2-70b-chat` (Longer responses)
- `gemma-7b-it` (Lighter)

## Rate Limits
Groq's free tier is generous:
- 30 requests per minute
- 6,000 requests per day

Perfect for thesis work and testing!

## Troubleshooting

**"Invalid API Key" error:**
- Verify the key starts with `gsk_`
- Check for extra spaces when copying
- Regenerate the key at https://console.groq.com/keys

**API errors:**
- Groq servers are usually up (99.9% uptime)
- Check your internet connection
- Try a different model if one fails

**Want to use a different AI service?**
You can swap the Groq module for:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- HuggingFace Inference API
- Local LLMs (Ollama, LM Studio)

Just create a new `summarizer.py` module with the same interface!

## Cost
**Free tier:** $0 (no credit card needed)

If you need higher limits later, Groq's paid plans start at $1/month.

---

Happy summarizing! 🚀
