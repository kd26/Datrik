from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime
import openai
import anthropic
import os
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
app.secret_key = os.environ.get('SECRET_KEY', 'datrik-secret-key-2025')

class DatrikAnalyst:
    def __init__(self, db_path: str = 'data/datrik.db'):
        self.db_path = db_path
        
        # Initialize AI clients
        self.openai_client = None
        self.anthropic_client = None
        
        # Try to initialize Anthropic (Primary)
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key and anthropic_key != 'your_anthropic_api_key_here':
            try:
                self.anthropic_client = anthropic.Client(api_key=anthropic_key)
            except Exception as e:
                print(f"Anthropic initialization failed: {e}")
        
        # Try to initialize OpenAI (Backup)
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and openai_key != 'your_openai_api_key_here':
            try:
                openai.api_key = openai_key
                self.openai_client = True  # Flag to indicate OpenAI is available
            except Exception as e:
                print(f"OpenAI initialization failed: {e}")
        
        # Fallback SQL templates
        self.fallback_queries = {
            'top restaurants': """
                SELECT r.name, r.cuisine_type, COUNT(o.order_id) as order_count, 
                       AVG(o.total_amount) as avg_order_value
                FROM restaurants r
                LEFT JOIN orders o ON r.restaurant_id = o.restaurant_id
                WHERE o.order_status = 'delivered'
                GROUP BY r.restaurant_id, r.name, r.cuisine_type
                ORDER BY order_count DESC
                LIMIT 10
            """,
            'popular cuisines': """
                SELECT r.cuisine_type, COUNT(o.order_id) as total_orders,
                       AVG(o.total_amount) as avg_order_value,
                       COUNT(DISTINCT o.user_id) as unique_customers
                FROM restaurants r
                JOIN orders o ON r.restaurant_id = o.restaurant_id
                WHERE o.order_status = 'delivered'
                GROUP BY r.cuisine_type
                ORDER BY total_orders DESC
                LIMIT 10
            """,
            'recent orders': """
                SELECT DATE(o.order_date) as order_day,
                       COUNT(o.order_id) as daily_orders,
                       AVG(o.total_amount) as avg_order_value,
                       SUM(o.total_amount) as daily_revenue
                FROM orders o
                WHERE o.order_status = 'delivered'
                  AND o.order_date >= DATE('now', '-7 days')
                GROUP BY DATE(o.order_date)
                ORDER BY order_day DESC
            """,
        }
        
        # Database schema
        self.schema_info = """
        Database Schema:
        1. users: user_id, email, first_name, last_name, registration_date, city, is_active, total_orders, lifetime_value, preferred_cuisine
        2. restaurants: restaurant_id, name, cuisine_type, rating, address, city, delivery_fee, minimum_order, is_active
        3. orders: order_id, user_id, restaurant_id, courier_id, order_date, order_status, subtotal, delivery_fee, tip_amount, total_amount, payment_method
        4. order_items: order_item_id, order_id, item_name, item_price, quantity, category
        5. couriers: courier_id, first_name, last_name, vehicle_type, rating, total_deliveries, is_active
        6. sessions: session_id, user_id, session_date, session_type, restaurant_id, device_type
        7. marketing_events: event_id, user_id, event_date, event_type, campaign_name, coupon_code, channel
        8. reviews: review_id, order_id, user_id, restaurant_id, food_rating, delivery_rating, overall_rating, review_date
        """
    
    def execute_query(self, query: str) -> dict:
        """Execute SQL query and return results as dict"""
        try:
            conn = sqlite3.connect(':memory:')  # Use in-memory for demo
            cursor = conn.cursor()
            cursor.execute(query)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Get all rows
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            conn.close()
            return {
                'data': results,
                'columns': columns,
                'row_count': len(results)
            }
        except Exception as e:
            print(f"Database error: {str(e)}")
            return {'data': [], 'columns': [], 'row_count': 0}
    
    def natural_language_to_sql(self, question: str) -> tuple[str, str]:
        """Convert natural language question to SQL"""
        system_prompt = f"""
        You are an expert SQL analyst for a food delivery app called Datrik. 
        Convert natural language questions into SQLite queries.
        
        {self.schema_info}
        
        Guidelines:
        1. Use proper SQLite syntax
        2. Use appropriate date functions
        3. Include ORDER BY clauses
        4. Use JOINs appropriately
        5. Return only the SQL query, no explanations
        6. Use LIMIT when appropriate
        
        Current date: {datetime.now().strftime('%Y-%m-%d')}
        """
        
        # Try Anthropic first (PRIMARY)
        if self.anthropic_client:
            try:
                response = self.anthropic_client.completion(
                    model="claude-2",
                    prompt=f"\n\nHuman: {system_prompt}\n\nQuestion: {question}\n\nSQL Query:\n\nAssistant:",
                    max_tokens_to_sample=500,
                    temperature=0.1
                )
                
                sql_query = response.completion.strip()
                sql_query = re.sub(r'```sql\n?', '', sql_query)
                sql_query = re.sub(r'```\n?', '', sql_query)
                
                return sql_query, "Anthropic (Primary)"
                
            except Exception as e:
                print(f"Anthropic failed: {e}")
        
        # Try OpenAI as backup
        if self.openai_client:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                sql_query = response.choices[0].message.content.strip()
                sql_query = re.sub(r'```sql\n?', '', sql_query)
                sql_query = re.sub(r'```\n?', '', sql_query)
                
                return sql_query, "OpenAI (Backup)"
                
            except Exception as e:
                print(f"OpenAI failed: {e}")
        
        # Fallback to templates
        question_lower = question.lower()
        for key, query in self.fallback_queries.items():
            if any(keyword in question_lower for keyword in key.split()):
                return query.strip(), "Template"
        
        return "SELECT 'No AI available' as message", "Error"
    
    def analyze_results(self, question: str, result_data: dict, provider: str) -> str:
        """Analyze query results"""
        data = result_data['data']
        columns = result_data['columns']
        row_count = result_data['row_count']
        
        if row_count == 0:
            return "No data found for your query."
        
        # Simple analysis for web deployment
        analysis = f"**Query Results ({provider}):**\n\n"
        analysis += f"Found {row_count:,} records.\n\n"
        
        # Show top 3 results
        analysis += "**Top Results:**\n"
        for i, row in enumerate(data[:3], 1):
            analysis += f"{i}. "
            for col in columns[:3]:
                analysis += f"{col}: {row.get(col, 'N/A')} | "
            analysis = analysis.rstrip("| ") + "\n"
        
        return analysis

# Initialize Datrik
datrik = DatrikAnalyst()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Generate SQL
        sql_query, provider = datrik.natural_language_to_sql(question)
        
        # Execute query
        result_data = datrik.execute_query(sql_query)
        
        # Analyze results
        analysis = datrik.analyze_results(question, result_data, provider)
        
        return jsonify({
            'sql_query': sql_query,
            'provider': provider,
            'analysis': analysis,
            'data': result_data['data'],
            'row_count': result_data['row_count']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    try:
        stats_data = {}
        conn = sqlite3.connect(datrik.db_path)
        
        tables = ['users', 'restaurants', 'orders', 'order_items', 'couriers', 'sessions', 'marketing_events', 'reviews']
        
        for table in tables:
            try:
                result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats_data[table] = result[0] if result else 0
            except:
                stats_data[table] = 0
        
        conn.close()
        return jsonify(stats_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For local development
if __name__ == '__main__':
    app.run(debug=True)