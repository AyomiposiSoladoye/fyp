@echo off
echo.
echo ============================================
echo Twitter Virality Prediction System
echo Fogg Behavior Model
echo ============================================
echo.
echo Starting Streamlit app...
echo Opening at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
streamlit run streamlit_app.py

pause
