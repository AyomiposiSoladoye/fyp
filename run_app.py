#!/usr/bin/env python
"""
Launcher script for the Streamlit app
"""

import subprocess
import sys
import os
import webbrowser
import time

def main():
    print("\n" + "="*60)
    print("Twitter Virality Prediction System - Streamlit App")
    print("Fogg Behavior Model Framework")
    print("="*60 + "\n")
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if required files exist
    if not os.path.exists('tweet_content-engagement_dataset.csv'):
        print("❌ Error: tweet_content-engagement_dataset.csv not found!")
        print("   Please make sure the CSV file is in the project directory.")
        sys.exit(1)
    
    if not os.path.exists('streamlit_app.py'):
        print("❌ Error: streamlit_app.py not found!")
        sys.exit(1)
    
    print("✓ Project files found")
    print("✓ Starting Streamlit server...\n")
    
    # Start Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nStreamlit server stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
