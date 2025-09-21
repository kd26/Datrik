#!/usr/bin/env python3
"""
Simple test to verify the Datrik app components work
"""

import sys
import os
sys.path.append('src')

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import sqlite3
        print("✅ sqlite3")
        
        import pandas as pd
        print("✅ pandas")
        
        import streamlit as st
        print("✅ streamlit")
        
        import openai
        print("✅ openai")
        
        from dotenv import load_dotenv
        print("✅ python-dotenv")
        
        import plotly.express as px
        print("✅ plotly")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_database():
    """Test database connectivity"""
    print("\n🗄️  Testing database...")
    
    try:
        import sqlite3
        conn = sqlite3.connect('data/datrik.db')
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        conn.close()
        
        print(f"✅ Database connected - {user_count:,} users found")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_openai_key():
    """Test OpenAI API key"""
    print("\n🔑 Testing OpenAI API key...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key.startswith('sk-'):
            print("✅ OpenAI API key found and formatted correctly")
            return True
        else:
            print("❌ OpenAI API key missing or invalid format")
            return False
    except Exception as e:
        print(f"❌ Error checking API key: {e}")
        return False

def main():
    print("🍕 DATRIK SYSTEM TEST")
    print("=" * 30)
    
    tests = [
        ("Module imports", test_imports),
        ("Database connectivity", test_database),
        ("OpenAI API key", test_openai_key)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 30)
    print(f"✅ Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All systems ready! You can now run:")
        print("   ./run_datrik.sh")
        print("   OR")
        print("   python3 -m streamlit run src/datrik_chat.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)