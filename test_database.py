#!/usr/bin/env python3
"""
Quick test script to verify the database was created correctly
"""

import sqlite3
import os

def test_database():
    """Test database connectivity and data"""
    db_path = 'data/datrik.db'
    
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Test basic queries
        tables = ['users', 'restaurants', 'orders', 'order_items', 'couriers', 'sessions', 'marketing_events', 'reviews']
        
        print("📊 Database Statistics:")
        print("-" * 30)
        
        for table in tables:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"{table:<20}: {count:>8,} records")
        
        # Test a sample query
        print("\n🔍 Sample Query Test:")
        print("-" * 30)
        
        query = """
        SELECT r.name, r.cuisine_type, COUNT(o.order_id) as order_count
        FROM restaurants r
        LEFT JOIN orders o ON r.restaurant_id = o.restaurant_id
        WHERE o.order_status = 'delivered'
        GROUP BY r.restaurant_id, r.name, r.cuisine_type
        ORDER BY order_count DESC
        LIMIT 5
        """
        
        cursor = conn.execute(query)
        results = cursor.fetchall()
        
        print("Top 5 restaurants by order volume:")
        for i, (name, cuisine, orders) in enumerate(results, 1):
            print(f"{i}. {name} ({cuisine}) - {orders} orders")
        
        conn.close()
        print("\n✅ Database test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    test_database()