# Tweet Examples to Test Virality Predictions

## HIGH VIRALITY POTENTIAL (Should predict Viral)

### Example 1: High Motivation + High Prompt
```
Just discovered this AMAZING new feature and it's absolutely life-changing!!! 🚀✨ 
Click here to see: http://example.com #Innovation #MustSee #Trending
```
**Why:** Emotional language, emojis, URL, hashtags, exclamation marks

---

### Example 2: High Ability + High Prompt
```
Elections coming up. Make sure you're registered to vote!
```
**Why:** Clear message, direct call-to-action, important topic

---

### Example 3: High Motivation + High Ability
```
Can't believe how far technology has come! The potential is HUGE. 😍
```
**Why:** Emotional, short, clear, enthusiastic

---

### Example 4: Multiple Hashtags + Questions
```
What's the best productivity app you've ever used? 
Looking to switch! #ProductivityHacks #Technology
```
**Why:** Question (prompts interaction), hashtags, clear intent

---

### Example 5: Viral Moment + Emojis
```
BREAKING: Something incredible just happened and I had to share! 🔥🔥🔥
You won't believe what happens next #Viral #Breaking
```
**Why:** Capital letters, emojis, urgency, excitement

---

## MEDIUM VIRALITY POTENTIAL (Should predict borderline)

### Example 6: Moderate Emotional Content
```
Traveling solo really opens your mind. #Wanderlust
```
**Why:** Some emotion, one hashtag, short

---

### Example 7: Straightforward News
```
New study shows coffee has more health benefits than previously thought.
```
**Why:** Informative, but no emotional hooks or CTAs

---

### Example 8: Mixed Signals
```
Layering is key this fall 🍂 #FashionTips
```
**Why:** One emoji, one hashtag, clear but simple

---

## LOW VIRALITY POTENTIAL (Should predict Non-Viral)

### Example 9: Bland & Unemotional
```
Had a long day at work. Really tired.
```
**Why:** No emotion, no hashtags, no calls to action

---

### Example 10: Unclear & Negative
```
I guess crypto might matter someday. Who knows.
```
**Why:** Passive, uncertain, negative undertone, no engagement elements

---

### Example 11: Too Generic
```
Another day, another meeting.
```
**Why:** No substance, no hooks, no reason to share

---

### Example 12: Pessimistic
```
Everything is getting worse these days.
```
**Why:** Negative sentiment, no solutions, no engagement elements

---

## EDGE CASES (Interesting to Test)

### Example 13: Only Emojis + Text
```
OMG THIS IS ABSOLUTELY INCREDIBLE!!! 🚀🚀🚀 #Amazing
```
**Why:** Very high motivation (all caps, exclamation), moderate prompt

---

### Example 14: Lots of Hashtags
```
Check this out! #Tech #Innovation #Startup #Business #Growth #Success #Trending
```
**Why:** Many hashtags (high prompt), but short content

---

### Example 15: Long Form + Multiple CTAs
```
I spent 3 months studying this topic and here's what I learned. 
The results surprised even me. Read the full analysis: http://example.com
Don't miss out! #Learning #Data
```
**Why:** Longer, multiple CTAs, credibility, emotion

---

### Example 16: Question Format
```
What's one thing you wish you'd learned earlier in life?
```
**Why:** Direct question (high prompt), engagement-focused

---

### Example 17: Controversial (High Engagement)
```
Hot take: Everything you think you know about productivity is WRONG 🔥
Here's why: http://example.com #Controversial #Debate
```
**Why:** Provocative, emojis, URL, hashtags

---

### Example 18: Celebration/Milestone
```
Just hit 1 MILLION followers!!! Thank you all so much! 🎉✨
This wouldn't be possible without YOU! #Grateful #Milestone
```
**Why:** Celebratory, exclamation marks, emojis, mentions gratitude

---

## TESTING STRATEGY

### Quick Test (5 examples):
1. Example 1 - High viral potential
2. Example 9 - Low viral potential
3. Example 6 - Medium potential
4. Example 13 - High motivation
5. Example 16 - High prompt

### Thorough Test (all 18):
Test all and observe patterns in which features drive predictions.

---

## What to Observe

**Check if predictions align with:**
- ✅ Motivation: Emotional words, emojis, capitals, punctuation
- ✅ Ability: Length, clarity, readability
- ✅ Prompt: Hashtags, mentions, questions, CTAs

**Examples that should be VIRAL:**
- 1, 2, 3, 4, 5, 13, 17, 18

**Examples that should be NON-VIRAL:**
- 9, 10, 11, 12

**Borderline (interesting to see):**
- 6, 7, 8, 14, 15, 16

---

## How to Test in Streamlit

1. Go to **🔮 Make Predictions** page
2. Paste each example
3. Check:
   - Viral or Non-Viral prediction?
   - Probability score?
   - Feature breakdown by dimension
4. Compare results across examples

This will show you which Fogg dimensions matter most! 🚀
