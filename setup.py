#!/usr/bin/env python3
"""
Datrik Setup Script
Automates the setup process for Datrik AI Food Delivery Analyst
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is sufficient"""
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print(f"Output: {e.output}")
        return False

def setup_environment():
    """Set up environment variables file"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        response = input("Do you want to overwrite it? (y/n): ").lower()
        if response != 'y':
            print("📝 Keeping existing .env file")
            return True
    
    if env_example.exists():
        # Copy .env.example to .env
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ Created .env file from template")
    else:
        # Create basic .env file
        with open(env_file, 'w') as f:
            f.write("""# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_PATH=./data/datrik.db

# Data Generation Configuration
RANDOM_SEED=42
DAYS_OF_DATA=90
""")
        print("✅ Created .env file with defaults")
    
    print("🔑 Please edit the .env file and add your OpenAI API key!")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["data", "csv_output"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True

def generate_data():
    """Generate sample data"""
    print("🎲 Generating sample data...")
    print("This may take a few minutes...")
    
    try:
        result = subprocess.run([sys.executable, "src/data_generator.py"], 
                               capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Sample data generated successfully")
            print("Generated files:")
            print("  - data/datrik.db (SQLite database)")
            print("  - csv_output/*.csv (CSV exports)")
            return True
        else:
            print(f"❌ Error generating data: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Data generation timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return False

def verify_setup():
    """Verify the setup is complete"""
    print("🔍 Verifying setup...")
    
    # Check if database exists
    if not Path("data/datrik.db").exists():
        print("❌ Database file not found")
        return False
    
    # Check if .env exists
    if not Path(".env").exists():
        print("❌ .env file not found")
        return False
    
    # Check if OpenAI key is set
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key or openai_key == 'your_openai_api_key_here':
            print("⚠️  OpenAI API key not set in .env file")
            print("   Please edit .env and add your OpenAI API key")
            return False
        else:
            print("✅ OpenAI API key found in .env")
            
    except ImportError:
        print("⚠️  Could not verify OpenAI key (python-dotenv not installed)")
    
    print("✅ Setup verification complete")
    return True

def main():
    """Main setup function"""
    print("🍕 Welcome to Datrik Setup!")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Setting up environment", setup_environment),
        ("Generating sample data", generate_data),
        ("Verifying setup", verify_setup)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            print(f"❌ Setup failed at step: {step_name}")
            return False
    
    print("\n" + "=" * 50)
    print("🎉 Datrik setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run: streamlit run src/datrik_chat.py")
    print("3. Open your browser and start asking questions!")
    print("\nExample questions to try:")
    print("  - Which restaurant had the highest orders last week?")
    print("  - What's the average order size by day of week?")
    print("  - Show me our top customers by lifetime value")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)