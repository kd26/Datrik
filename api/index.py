from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime
import openai
import anthropic
import os
from dotenv import load_dotenv
import json
import re
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from langchain_analyst import LangChainDatrikAnalyst

# Load environment variables
load_dotenv()

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
app.secret_key = os.environ.get('SECRET_KEY', 'datrik-secret-key-2025')

class DatrikAnalyst:
    def __init__(self, db_path: str = None):
        # Set correct database path for Vercel deployment
        if db_path is None:
            # Try different possible paths
            possible_paths = [
                'data/datrik.db',
                '../data/datrik.db', 
                '/var/task/data/datrik.db',
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'datrik.db')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    db_path = path
                    break
            else:
                db_path = 'data/datrik.db'  # fallback
        self.db_path = db_path
        
        # Initialize AI clients
        self.openai_client = None
        self.anthropic_client = None
        
        # Try to initialize Anthropic (Primary)
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        print(f"Debug: ANTHROPIC_API_KEY found: {bool(anthropic_key)}")
        if anthropic_key and anthropic_key != 'your_anthropic_api_key_here':
            print(f"Debug: Attempting Anthropic initialization...")
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                print(f"Debug: Anthropic client initialized successfully")
            except Exception as e:
                print(f"Anthropic initialization failed: {e}")
        else:
            print(f"Debug: Anthropic key not found or is placeholder")
        
        # Try to initialize OpenAI (Backup)
        openai_key = os.getenv('OPENAI_API_KEY')
        print(f"Debug: OPENAI_API_KEY found: {bool(openai_key)}")
        if openai_key and openai_key != 'your_openai_api_key_here':
            print(f"Debug: Attempting OpenAI initialization...")
            try:
                openai.api_key = openai_key
                self.openai_client = True  # Flag to indicate OpenAI is available
                print(f"Debug: OpenAI client initialized successfully")
            except Exception as e:
                print(f"OpenAI initialization failed: {e}")
        else:
            print(f"Debug: OpenAI key not found or is placeholder")
            
        print(f"Debug: Final state - Anthropic: {bool(self.anthropic_client)}, OpenAI: {bool(self.openai_client)}")
        
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
            # Try to connect to the actual database file
            conn = sqlite3.connect(self.db_path)
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
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=500,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": f"Question: {question}\n\nPlease provide only the SQL query, no explanations."}
                    ]
                )
                
                sql_query = response.content[0].text.strip()
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
    
    def generate_ai_insights(self, question: str, sql_query: str, result_data: dict, provider: str, conversation_history: list = None) -> str:
        """Use Anthropic to generate conversational insights from data"""
        data = result_data['data']
        row_count = result_data['row_count']
        
        if row_count == 0:
            return "I couldn't find any data matching your query. Try asking about something else!"
        
        # Prepare data sample for AI context (limit to first 10 rows for token efficiency)
        data_sample = data[:10] if data else []
        
        # Build conversation context
        context_section = ""
        if conversation_history and len(conversation_history) > 0:
            recent_context = conversation_history[-3:]  # Last 3 exchanges for context
            context_section = f"\nPrevious conversation context:\n"
            for i, exchange in enumerate(recent_context):
                context_section += f"Q{i+1}: {exchange.get('question', '')}\nA{i+1}: {exchange.get('response', '')[:200]}...\n"
            context_section += f"\nNote: If the current question refers to 'above data', 'previous data', 'that data', or similar references, use the previous conversation context.\n"

        # Create rich context prompt for AI
        context_prompt = f"""You are a super smart analyst. Give me the key insights from this data in simple, human language.

Current question: "{question}"
Data found: {row_count:,} records
Current results: {json.dumps(data_sample, indent=2) if data_sample else "No data available"}{context_section}

Response format:
- Start with a quick answer to their question
- Give 2-3 bullet points with the most important takeaways
- End with 1 actionable recommendation

Rules:
- Use simple English, no jargon or fancy words
- Keep it short and punchy
- Use **bold** for key numbers and insights
- Focus on what matters for business decisions
- No fluff or corporate speak
- Maximum 3-4 sentences total
- If question refers to previous data/context, acknowledge and use it"""

        # Try AI-powered analysis first
        if self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=300,
                    temperature=0.2,
                    system="You are a super smart analyst. Give concise, actionable insights in simple language. No corporate fluff.",
                    messages=[{"role": "user", "content": context_prompt}]
                )
                return response.content[0].text.strip()
            except Exception as e:
                print(f"AI insight generation failed: {e}")
        
        # Fallback to basic analysis if AI fails
        return self.basic_analysis(result_data)
    
    def basic_analysis(self, result_data: dict) -> str:
        """Fallback analysis when AI is unavailable"""
        data = result_data['data']
        row_count = result_data['row_count']
        
        if not data:
            return f"Found {row_count:,} records, but no data to display."
        
        # Simple fallback based on data structure
        first_row = data[0]
        if 'name' in first_row and 'total_orders' in first_row:
            top_name = first_row['name']
            top_orders = first_row['total_orders']
            return f"Found {row_count:,} restaurants. **{top_name}** leads with {top_orders:,} orders."
        elif 'cuisine_type' in first_row:
            top_cuisine = first_row['cuisine_type']
            return f"Analyzed {row_count:,} cuisine types. **{top_cuisine}** appears to be the top performer."
        else:
            return f"Found {row_count:,} records matching your query. Here are the results:"
    
    def analyze_results(self, question: str, result_data: dict, provider: str, sql_query: str = "", conversation_history: list = None) -> str:
        """Generate AI-powered conversational analysis of query results"""
        return self.generate_ai_insights(question, sql_query, result_data, provider, conversation_history)

# Initialize LangChain Datrik Analyst
datrik = LangChainDatrikAnalyst()

# Keep the old class for fallback
class DatrikAnalyst:
    def __init__(self, db_path: str = None):
        if db_path is None:
            possible_paths = [
                'data/datrik.db',
                '../data/datrik.db', 
                '/var/task/data/datrik.db',
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'datrik.db')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    db_path = path
                    break
            else:
                db_path = 'data/datrik.db'
        self.db_path = db_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        question = data.get('question', '')
        session_id = data.get('session_id', None)
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Use LangChain analyst with memory
        try:
            analysis, result_data, provider = datrik.analyze_with_memory(question, session_id)
            
            # Get SQL query from the last operation (for display purposes)
            sql_query = "Generated with LangChain"  # We'll enhance this later
            
            return jsonify({
                'sql_query': sql_query,
                'provider': provider,
                'analysis': analysis,
                'data': result_data['data'],
                'row_count': result_data['row_count'],
                'session_id': datrik.current_session_id
            })
            
        except Exception as e:
            print(f"LangChain analysis failed: {e}")
            # Fallback to original system
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions', methods=['GET', 'POST'])
def sessions():
    """Manage conversation sessions"""
    try:
        if request.method == 'POST':
            # Create new session
            data = request.get_json()
            user_identifier = data.get('user_identifier', 'default')
            session_id = datrik.start_new_session(user_identifier)
            
            return jsonify({
                'session_id': session_id,
                'message': 'New session created'
            })
        
        else:
            # Get recent sessions
            user_identifier = request.args.get('user_identifier', 'default')
            sessions = datrik.memory_manager.get_recent_sessions(user_identifier)
            
            return jsonify({'sessions': sessions})
            
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

@app.route('/api/debug')
def debug():
    """Debug endpoint to check environment variables and AI client status"""
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    return jsonify({
        'environment_variables': {
            'ANTHROPIC_API_KEY': f"{'✅ Set' if anthropic_key else '❌ Missing'} ({len(anthropic_key) if anthropic_key else 0} chars)",
            'OPENAI_API_KEY': f"{'✅ Set' if openai_key else '❌ Missing'} ({len(openai_key) if openai_key else 0} chars)"
        },
        'ai_clients': {
            'anthropic_initialized': bool(datrik.anthropic_client),
            'openai_initialized': bool(datrik.openai_client)
        },
        'database_path': datrik.db_path,
        'database_exists': os.path.exists(datrik.db_path)
    })

# For local development
if __name__ == '__main__':
    app.run(debug=True)