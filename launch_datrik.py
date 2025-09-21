#!/usr/bin/env python3
"""
Launch script for Datrik AI Analyst
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if database exists
    if not Path("data/datrik.db").exists():
        print("❌ Database not found! Please run: python src/data_generator.py")
        return False
    
    # Check if .env exists
    if not Path(".env").exists():
        print("❌ .env file not found! Please copy .env.example to .env and add your OpenAI API key")
        return False
    
    # Check if OpenAI key is set
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key or openai_key == 'your_openai_api_key_here':
            print("❌ OpenAI API key not set! Please edit .env file and add your API key")
            return False
    except ImportError:
        print("⚠️  Could not check OpenAI key (python-dotenv not installed)")
    
    print("✅ All requirements met!")
    return True

def launch_streamlit():
    """Launch the Streamlit app"""
    try:
        print("🚀 Launching Datrik...")
        print("📱 Your browser should open automatically")
        print("💡 If not, go to: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop Datrik")
        print("-" * 50)
        
        # Try to launch streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/datrik_chat.py"], check=True)
        
    except subprocess.CalledProcessError:
        print("❌ Failed to launch Streamlit")
        print("💡 Try running manually: streamlit run src/datrik_chat.py")
        return False
    except KeyboardInterrupt:
        print("\n👋 Datrik stopped. Thanks for using our AI analyst!")
        return True

def main():
    """Main launch function"""
    print("🍕 Welcome to Datrik - AI Food Delivery Analyst!")
    print("=" * 50)
    
    if not check_requirements():
        print("\n🔧 Setup incomplete. Please resolve the issues above.")
        return False
    
    return launch_streamlit()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)