import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import os

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

class ConversationMemoryManager:
    """Manages conversation memory with SQLite persistence"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_memory_tables()
    
    def init_memory_tables(self):
        """Initialize memory tables in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            # Read and execute memory schema
            schema_path = os.path.join(os.path.dirname(__file__), 'memory_schema.sql')
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            conn.executescript(schema_sql)
            conn.commit()
            conn.close()
            print("Memory tables initialized successfully")
        except Exception as e:
            print(f"Error initializing memory tables: {e}")
    
    def create_session(self, user_identifier: str = "default") -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO conversation_sessions (session_id, user_identifier)
                VALUES (?, ?)
            """, (session_id, user_identifier))
            conn.commit()
            conn.close()
            return session_id
        except Exception as e:
            print(f"Error creating session: {e}")
            return session_id
    
    def save_message(self, session_id: str, message_type: str, content: str, metadata: Dict = None):
        """Save a message to conversation memory"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO conversation_memory (session_id, message_type, content, metadata, token_count)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, message_type, content, json.dumps(metadata or {}), len(content.split())))
            
            # Update session last activity
            conn.execute("""
                UPDATE conversation_sessions 
                SET last_activity = CURRENT_TIMESTAMP, 
                    total_messages = total_messages + 1
                WHERE session_id = ?
            """, (session_id,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving message: {e}")
    
    def load_session_history(self, session_id: str, limit: int = 20) -> List[BaseMessage]:
        """Load conversation history for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT message_type, content FROM conversation_memory
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, (session_id, limit))
            
            messages = []
            for msg_type, content in cursor.fetchall():
                if msg_type == 'human':
                    messages.append(HumanMessage(content=content))
                elif msg_type == 'ai':
                    messages.append(AIMessage(content=content))
            
            conn.close()
            return messages
        except Exception as e:
            print(f"Error loading session history: {e}")
            return []
    
    def get_recent_sessions(self, user_identifier: str = "default", limit: int = 5) -> List[Dict]:
        """Get recent conversation sessions for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, start_time, last_activity, total_messages, conversation_summary
                FROM conversation_sessions
                WHERE user_identifier = ?
                ORDER BY last_activity DESC
                LIMIT ?
            """, (user_identifier, limit))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    'session_id': row[0],
                    'start_time': row[1],
                    'last_activity': row[2],
                    'total_messages': row[3],
                    'summary': row[4]
                })
            
            conn.close()
            return sessions
        except Exception as e:
            print(f"Error getting recent sessions: {e}")
            return []

class LangChainDatrikAnalyst:
    """Enhanced Datrik Analyst with LangChain memory capabilities"""
    
    def __init__(self, db_path: str = None):
        # Set database path
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
        self.memory_manager = ConversationMemoryManager(db_path)
        
        # Current session
        self.current_session_id = None
        
        # Database schema info for SQL generation
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
        
        # Initialize LangChain components after schema_info is defined
        self.setup_langchain()
    
    def setup_langchain(self):
        """Initialize LangChain components"""
        # Initialize default values
        self.memory = None
        self.llm = None
        self.conversation_chain = None
        self.langchain_available = False
        
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        if anthropic_key and anthropic_key != 'your_anthropic_api_key_here':
            try:
                # Initialize Anthropic LLM
                self.llm = ChatAnthropic(
                    model="claude-3-haiku-20240307",
                    temperature=0.2,
                    max_tokens=800,
                    anthropic_api_key=anthropic_key
                )
                
                # Create conversation memory (short-term)
                self.memory = ConversationBufferWindowMemory(
                    k=10,  # Keep last 10 exchanges
                    memory_key="chat_history",
                    return_messages=True
                )
                
                # Create the conversation prompt template
                self.prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are Datrik, a super smart food delivery analyst. You have conversation memory and can:

- Remember what the user asked before in this conversation
- Ask clarifying questions instead of starting fresh
- Reference previous queries and build on them
- Provide contextual insights based on conversation history

Current database context: {self.schema_info}

Guidelines:
- Use simple English, no jargon
- Give concise, actionable insights 
- Remember previous context and reference it when relevant
- Ask clarifying questions when needed
- If the user refers to "previous", "earlier", "that data" etc., use conversation memory
- Maximum 3-4 sentences for analysis
- Focus on business insights that matter"""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                
                # Create the conversation chain
                self.conversation_chain = LLMChain(
                    llm=self.llm,
                    prompt=self.prompt,
                    memory=self.memory,
                    verbose=True
                )
                
                self.langchain_available = True
                print("LangChain setup successful")
                
            except Exception as e:
                print(f"LangChain setup failed: {e}")
                self.langchain_available = False
        else:
            print("No Anthropic API key found")
            self.langchain_available = False
    
    def start_new_session(self, user_identifier: str = "default") -> str:
        """Start a new conversation session"""
        self.current_session_id = self.memory_manager.create_session(user_identifier)
        # Reset in-memory conversation buffer if available
        if self.memory:
            self.memory.clear()
        return self.current_session_id
    
    def load_session(self, session_id: str):
        """Load existing conversation session"""
        self.current_session_id = session_id
        
        # Load conversation history into memory if available
        if self.memory:
            messages = self.memory_manager.load_session_history(session_id)
            
            # Clear current memory and add historical messages
            self.memory.clear()
            for message in messages:
                if isinstance(message, HumanMessage):
                    self.memory.chat_memory.add_user_message(message.content)
                elif isinstance(message, AIMessage):
                    self.memory.chat_memory.add_ai_message(message.content)
    
    def execute_query(self, query: str) -> dict:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
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
    
    def analyze_with_memory(self, question: str, session_id: str = None) -> Tuple[str, dict, str]:
        """Analyze question with conversation memory"""
        
        # Use provided session or current session
        if session_id:
            if session_id != self.current_session_id:
                self.load_session(session_id)
        elif not self.current_session_id:
            self.start_new_session()
        
        # Save user question to persistent memory
        self.memory_manager.save_message(
            self.current_session_id, 
            'human', 
            question,
            {'timestamp': datetime.now().isoformat()}
        )
        
        if not self.langchain_available:
            return "LangChain not available", {}, "Error"
        
        try:
            # First, generate SQL query (this part stays similar)
            sql_query, provider = self.generate_sql(question)
            
            # Execute the query
            result_data = self.execute_query(sql_query)
            
            # Use LangChain for contextual analysis
            analysis_input = {
                'question': question,
                'schema_info': self.schema_info,
                'query_results': f"SQL Query: {sql_query}\nResults: {result_data['row_count']} records found\nSample data: {json.dumps(result_data['data'][:3], indent=2)}"
            }
            
            # Get contextual analysis from LangChain
            response = self.conversation_chain.invoke({
                "input": f"{question}\n\nQuery executed: {sql_query}\nResults: {result_data['row_count']} records found\nSample data: {json.dumps(result_data['data'][:3], indent=2) if result_data['data'] else 'No data'}"
            })["text"]
            
            # Check if AI recommends a different/additional query
            additional_query = self.extract_recommended_query(response)
            if additional_query and additional_query != sql_query.strip():
                print(f"AI recommended additional query, executing: {additional_query[:100]}...")
                additional_results = self.execute_query(additional_query)
                if additional_results['row_count'] > 0:
                    # Update response with actual results
                    response += f"\n\n**Updated Analysis with Recommended Query:**\n"
                    response += self.generate_insights_from_data(additional_results, question)
                    result_data = additional_results  # Use the recommended query results
            
            # Save AI response to persistent memory
            self.memory_manager.save_message(
                self.current_session_id,
                'ai',
                response,
                {
                    'sql_query': sql_query,
                    'result_count': result_data['row_count'],
                    'provider': provider
                }
            )
            
            return response, result_data, provider, sql_query
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(error_msg)
            return error_msg, {'data': [], 'columns': [], 'row_count': 0}, "Error", ""
    
    def generate_sql(self, question: str) -> Tuple[str, str]:
        """Generate SQL query using LLM with conversation context"""
        
        # Try LLM-based SQL generation first
        if self.langchain_available:
            return self.generate_sql_with_llm(question)
        else:
            return self.intelligent_fallback(question)
    
    def generate_sql_with_llm(self, question: str) -> Tuple[str, str]:
        """Generate SQL using LLM with database schema context"""
        
        # Check conversation history for context and extract previous recommendations
        context_info = ""
        previous_recommendations = ""
        if self.memory and hasattr(self.memory, 'chat_memory'):
            recent_messages = self.memory.chat_memory.messages[-4:]  # Last 2 exchanges
            if recent_messages:
                context_info = "\nRecent conversation context:\n"
                for i, msg in enumerate(recent_messages):
                    role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                    context_info += f"{role}: {msg.content[:100]}...\n"
                    
                    # Extract previous recommendations from AI messages
                    if role == "Assistant" and ("recommend" in msg.content or "suggest" in msg.content or "points" in msg.content):
                        previous_recommendations += f"Previous recommendation: {msg.content}\n"
        
        # Detect if this is a follow-up to previous recommendations
        is_followup_request = any(phrase in question.lower() for phrase in [
            'second point', 'do second', 'point 2', 'that you recommended', 'you suggested',
            'cuisine', 'by cuisine', 'breakdown', 'dimension', 'categories'
        ])
        
        # Special handling for follow-up requests
        if is_followup_request:
            if any(word in question.lower() for word in ['second', 'point 2', 'cuisine', 'breakdown', 'dimension']):
                # Generate cuisine-based monthly analysis
                return """
                SELECT 
                    DATE(o.order_date, 'start of month') AS order_month,
                    r.cuisine_type,
                    COUNT(o.order_id) AS total_orders,
                    SUM(o.total_amount) AS total_revenue,
                    AVG(o.total_amount) AS avg_order_value
                FROM orders o
                JOIN restaurants r ON o.restaurant_id = r.restaurant_id
                WHERE o.order_status = 'delivered'
                GROUP BY order_month, r.cuisine_type
                ORDER BY order_month DESC, total_orders DESC
                LIMIT 50
                """.strip(), "LLM Generated (Follow-up)"

        prompt = f"""You are a SQL expert for a food delivery database. Generate a valid SQLite query for the user's question.

Database Schema:
{self.schema_info}

{context_info}
{previous_recommendations}

User Question: {question}

Rules:
1. ONLY return the SQL query, no explanations
2. Use proper SQLite syntax and functions
3. Include meaningful JOINs when needed
4. Add LIMIT clauses for large results (typically 10-20 records)
5. Use aliases for readability
6. Filter for 'delivered' status when querying orders
7. If question refers to previous data/context, enhance the query with additional dimensions
8. For follow-up questions, expand the analysis to include more relevant columns
9. When user asks about trends or comparisons, include GROUP BY with multiple dimensions

Enhanced Query Guidelines:
- For "weekly trends" → GROUP BY DATE(order_date, 'weekday 0', '-6 days') to aggregate by week start
- For "daily trends" → GROUP BY DATE(order_date) for individual days
- For "monthly trends" → GROUP BY DATE(order_date, 'start of month') for monthly aggregation
- For "cuisine analysis" → JOIN with restaurants table for cuisine_type
- For "by categories" → Always GROUP BY the category dimension
- For comparisons → Include comparative metrics (AVG, COUNT, SUM)

Time Period Examples:
- Weekly: SELECT DATE(order_date, 'weekday 0', '-6 days') as week_start, COUNT(*) as weekly_orders
- Daily: SELECT DATE(order_date) as order_date, COUNT(*) as daily_orders
- Monthly: SELECT DATE(order_date, 'start of month') as month_start, COUNT(*) as monthly_orders

SQL Query:"""

        try:
            response = self.llm.invoke(prompt)
            sql_query = response.content.strip()
            
            # Clean up the response
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            # Validate and sanitize
            sql_query, is_safe = self.validate_and_sanitize_sql(sql_query)
            
            if is_safe and sql_query.upper().startswith('SELECT'):
                return sql_query, "LLM Generated"
            else:
                return self.intelligent_fallback(question)
                
        except Exception as e:
            print(f"LLM SQL generation failed: {e}")
            return self.intelligent_fallback(question)
    
    def validate_and_sanitize_sql(self, sql: str) -> Tuple[str, bool]:
        """Validate SQL query for safety and correctness"""
        
        # Basic safety checks
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        sql_upper = sql.upper()
        
        if any(keyword in sql_upper for keyword in dangerous_keywords):
            return "SELECT 'Query contains unsafe operations' as error", False
        
        # Check for basic SQL structure
        if not sql.strip().upper().startswith('SELECT'):
            return "SELECT 'Invalid query format' as error", False
        
        # Basic injection protection
        if any(dangerous in sql_upper for dangerous in ['--', '/*', '*/', ';DROP', ';DELETE']):
            return "SELECT 'Potentially malicious query detected' as error", False
        
        return sql, True
    
    def intelligent_fallback(self, question: str) -> Tuple[str, str]:
        """Generate fallback query using pattern matching and schema analysis"""
        
        question_lower = question.lower()
        
        # Extract intent and entities
        if any(word in question_lower for word in ['top', 'best', 'most popular']) and any(word in question_lower for word in ['restaurant', 'place']):
            sql = """
                SELECT r.name, r.cuisine_type, COUNT(o.order_id) as total_orders,
                       AVG(o.total_amount) as avg_order_value,
                       r.rating
                FROM restaurants r
                LEFT JOIN orders o ON r.restaurant_id = o.restaurant_id
                WHERE o.order_status = 'delivered'
                GROUP BY r.restaurant_id, r.name, r.cuisine_type, r.rating
                ORDER BY total_orders DESC
                LIMIT 10
            """
            return sql.strip(), "Fallback"
        
        elif any(word in question_lower for word in ['cuisine', 'food type', 'category']):
            sql = """
                SELECT r.cuisine_type, COUNT(o.order_id) as total_orders,
                       AVG(o.total_amount) as avg_order_value,
                       COUNT(DISTINCT r.restaurant_id) as restaurant_count
                FROM restaurants r
                JOIN orders o ON r.restaurant_id = o.restaurant_id
                WHERE o.order_status = 'delivered'
                GROUP BY r.cuisine_type
                ORDER BY total_orders DESC
                LIMIT 10
            """
            return sql.strip(), "Fallback"
        
        elif any(word in question_lower for word in ['weekly', 'week']):
            sql = """
                SELECT DATE(o.order_date, 'weekday 0', '-6 days') as week_start,
                       COUNT(o.order_id) as weekly_orders,
                       AVG(o.total_amount) as avg_order_value,
                       SUM(o.total_amount) as weekly_revenue
                FROM orders o
                WHERE o.order_status = 'delivered'
                  AND o.order_date >= DATE('now', '-8 weeks')
                GROUP BY week_start
                ORDER BY week_start DESC
                LIMIT 10
            """
            return sql.strip(), "Fallback"
        
        elif any(word in question_lower for word in ['monthly', 'month']):
            sql = """
                SELECT DATE(o.order_date, 'start of month') as month_start,
                       COUNT(o.order_id) as monthly_orders,
                       AVG(o.total_amount) as avg_order_value,
                       SUM(o.total_amount) as monthly_revenue
                FROM orders o
                WHERE o.order_status = 'delivered'
                  AND o.order_date >= DATE('now', '-6 months')
                GROUP BY month_start
                ORDER BY month_start DESC
                LIMIT 10
            """
            return sql.strip(), "Fallback"
        
        elif any(word in question_lower for word in ['order', 'trend', 'daily']):
            sql = """
                SELECT DATE(o.order_date) as order_day,
                       COUNT(o.order_id) as daily_orders,
                       AVG(o.total_amount) as avg_order_value,
                       SUM(o.total_amount) as daily_revenue
                FROM orders o
                WHERE o.order_status = 'delivered'
                  AND o.order_date >= DATE('now', '-7 days')
                GROUP BY DATE(o.order_date)
                ORDER BY order_day DESC
            """
            return sql.strip(), "Fallback"
        
        elif any(word in question_lower for word in ['user', 'customer', 'frequency']):
            sql = """
                SELECT u.city, AVG(u.total_orders) as avg_orders_per_user,
                       COUNT(u.user_id) as user_count,
                       AVG(u.lifetime_value) as avg_lifetime_value
                FROM users u
                WHERE u.is_active = 1
                GROUP BY u.city
                ORDER BY avg_orders_per_user DESC
                LIMIT 10
            """
            return sql.strip(), "Fallback"
        
        elif any(word in question_lower for word in ['average', 'avg', 'mean']) and any(word in question_lower for word in ['order', 'value', 'amount']):
            sql = """
                SELECT 'Overall average order value' as metric,
                       AVG(total_amount) as value,
                       COUNT(*) as total_orders
                FROM orders 
                WHERE order_status = 'delivered'
            """
            return sql.strip(), "Fallback"
        
        else:
            # Generic exploratory query
            sql = """
                SELECT 'Total restaurants' as metric, COUNT(*) as value FROM restaurants
                UNION ALL
                SELECT 'Total orders' as metric, COUNT(*) as value FROM orders WHERE order_status = 'delivered'
                UNION ALL
                SELECT 'Active users' as metric, COUNT(*) as value FROM users WHERE is_active = 1
                UNION ALL
                SELECT 'Average order value' as metric, ROUND(AVG(total_amount), 2) as value FROM orders WHERE order_status = 'delivered'
            """
            return sql.strip(), "Fallback"
    
    def extract_recommended_query(self, ai_response: str) -> str:
        """Extract SQL query from AI response if it recommends one"""
        try:
            # Look for SQL blocks in the response
            import re
            
            # Try to find SELECT statements in the response
            sql_patterns = [
                r'SELECT[\s\S]*?(?=\n\n|\n[A-Z]|\nThis|\nSome|\nLet|\n-|\nTo|\Z)',
                r'```sql\n(.*?)\n```',
                r'```\n(SELECT.*?)\n```'
            ]
            
            for pattern in sql_patterns:
                matches = re.findall(pattern, ai_response, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    query = match.strip() if isinstance(match, str) else match
                    if query.upper().startswith('SELECT') and len(query) > 20:
                        # Clean up the query
                        query = re.sub(r'^\s*```sql\s*\n?', '', query, flags=re.IGNORECASE)
                        query = re.sub(r'\n?\s*```\s*$', '', query, flags=re.IGNORECASE)
                        return query.strip()
            
            return None
        except Exception as e:
            print(f"Error extracting recommended query: {e}")
            return None
    
    def generate_insights_from_data(self, result_data: dict, original_question: str) -> str:
        """Generate quick insights from actual data results"""
        if not result_data['data'] or result_data['row_count'] == 0:
            return "No additional data found with the recommended query."
        
        data = result_data['data'][:5]  # First 5 records for insight
        
        try:
            # Create a quick summary based on the data structure
            if any('cuisine_type' in str(row).lower() for row in data):
                cuisines = [row.get('cuisine_type', '') for row in data if row.get('cuisine_type')]
                total_orders = sum([row.get('total_orders', 0) for row in data if row.get('total_orders')])
                return f"Found data for {len(set(cuisines))} different cuisine types with {total_orders:,} total orders analyzed by cuisine breakdown."
            
            elif any('month' in str(row).lower() for row in data):
                months = len(set([row.get('order_month', '') for row in data if row.get('order_month')]))
                return f"Monthly data showing trends across {months} months with detailed breakdowns."
            
            else:
                return f"Additional analysis shows {result_data['row_count']} records with enhanced data dimensions."
                
        except Exception as e:
            return f"Additional data retrieved with {result_data['row_count']} records for deeper analysis."