#!/usr/bin/env python3
"""
Project Summary for Datrik AI Analyst
"""

import os
from pathlib import Path

def print_project_summary():
    """Print comprehensive project summary"""
    
    print("🍕 DATRIK - AI FOOD DELIVERY ANALYST")
    print("=" * 60)
    
    print("\n📁 PROJECT STRUCTURE:")
    print("-" * 30)
    
    files_to_check = [
        ("src/database_schema.sql", "📋 Database Schema"),
        ("src/data_generator.py", "🎲 Data Generator"),
        ("src/datrik_chat.py", "💬 Chat Interface"),
        ("data/datrik.db", "🗄️  SQLite Database"),
        ("csv_output/", "📊 CSV Exports"),
        ("requirements.txt", "📦 Dependencies"),
        (".env.example", "⚙️  Environment Template"),
        ("README.md", "📖 Documentation"),
        ("setup.py", "🔧 Setup Script"),
        ("launch_datrik.py", "🚀 Launch Script"),
    ]
    
    for file_path, description in files_to_check:
        path = Path(file_path)
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                size_str = f"({size:,} bytes)" if size < 1024*1024 else f"({size/(1024*1024):.1f} MB)"
                print(f"✅ {description:<25} {size_str}")
            else:
                # Directory
                file_count = len(list(path.glob("*")))
                print(f"✅ {description:<25} ({file_count} files)")
        else:
            print(f"❌ {description:<25} (missing)")
    
    print("\n📊 DATABASE STATISTICS:")
    print("-" * 30)
    
    db_path = Path("data/datrik.db")
    if db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            
            tables = ['users', 'restaurants', 'orders', 'order_items', 'couriers', 'sessions', 'marketing_events', 'reviews']
            total_records = 0
            
            for table in tables:
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    total_records += count
                    print(f"📊 {table:<20}: {count:>8,} records")
                except:
                    print(f"❌ {table:<20}: error reading")
            
            print(f"{'='*30}")
            print(f"📊 {'TOTAL':<20}: {total_records:>8,} records")
            
            # Get date range
            try:
                cursor = conn.execute("SELECT MIN(order_date), MAX(order_date) FROM orders")
                result = cursor.fetchone()
                if result and result[0]:
                    start_date = result[0][:10]
                    end_date = result[1][:10]
                    print(f"📅 Data Range          : {start_date} to {end_date}")
            except:
                pass
            
            conn.close()
            
        except ImportError:
            print("⚠️  Cannot check database (sqlite3 not available)")
        except Exception as e:
            print(f"❌ Error reading database: {e}")
    else:
        print("❌ Database not found")
    
    print("\n🚀 GETTING STARTED:")
    print("-" * 30)
    print("1. Set up environment:")
    print("   cp .env.example .env")
    print("   # Edit .env and add your OpenAI API key")
    print()
    print("2. Generate data (if not done):")
    print("   python src/data_generator.py")
    print()
    print("3. Launch Datrik:")
    print("   python launch_datrik.py")
    print("   # OR")
    print("   streamlit run src/datrik_chat.py")
    print()
    
    print("💡 EXAMPLE QUESTIONS TO TRY:")
    print("-" * 30)
    examples = [
        "Which restaurant had the highest orders last week?",
        "What was the average order size this month?", 
        "Show me week over week order trends",
        "Which cuisine is most popular?",
        "What's our customer retention rate?",
        "How effective are our marketing campaigns?",
        "Which couriers have the highest ratings?",
        "What's the conversion rate from sessions to orders?"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i:2}. {example}")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("-" * 30)
    print("• Database not found → Run: python src/data_generator.py")
    print("• OpenAI API errors → Check your API key in .env file")
    print("• Module not found → Run: pip install -r requirements.txt")
    print("• Streamlit not found → Add Python bin to PATH or use 'python -m streamlit'")
    
    print("\n✅ PROJECT STATUS: READY TO USE!")
    print("=" * 60)

if __name__ == "__main__":
    print_project_summary()