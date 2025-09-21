import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import openai
import os
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()

class DatrikAnalyst:
    def __init__(self, db_path: str = 'data/datrik.db'):
        self.db_path = db_path
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Database schema for context
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
        
        Key relationships:
        - orders.user_id → users.user_id
        - orders.restaurant_id → restaurants.restaurant_id
        - orders.courier_id → couriers.courier_id
        - order_items.order_id → orders.order_id
        - reviews.order_id → orders.order_id
        """
    
    def get_database_connection(self):
        """Create database connection"""
        return sqlite3.connect(self.db_path)
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        try:
            conn = self.get_database_connection()
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return pd.DataFrame()
    
    def natural_language_to_sql(self, question: str) -> str:
        """Convert natural language question to SQL using OpenAI"""
        
        system_prompt = f"""
        You are an expert SQL analyst for a food delivery app called Datrik. 
        Convert natural language questions into SQLite queries.
        
        {self.schema_info}
        
        Important guidelines:
        1. Always use proper SQLite syntax
        2. Use appropriate date functions (DATE(), DATETIME(), strftime())
        3. Handle time-based queries carefully (last week, last month, etc.)
        4. For "last week", use the previous 7 days from current date
        5. For "week over week", compare same periods in different weeks
        6. Always include ORDER BY clauses for meaningful sorting
        7. Use JOINs appropriately to connect related tables
        8. Return only the SQL query, no explanations
        9. Use LIMIT when appropriate to avoid huge result sets
        
        Current date context: {datetime.now().strftime('%Y-%m-%d')}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the query (remove markdown formatting if present)
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            
            return sql_query
            
        except Exception as e:
            st.error(f"Error generating SQL: {str(e)}")
            return ""
    
    def analyze_results(self, question: str, query: str, df: pd.DataFrame) -> str:
        """Analyze query results and provide intelligent insights"""
        
        if df.empty:
            return "No data found for your query."
        
        # Convert DataFrame to a summary for analysis
        df_summary = {
            'row_count': len(df),
            'columns': list(df.columns),
            'sample_data': df.head(3).to_dict('records') if len(df) > 0 else [],
            'data_types': df.dtypes.to_dict()
        }
        
        analysis_prompt = f"""
        You are Datrik, an intelligent food delivery analyst. Analyze the SQL query results and provide meaningful insights.
        
        Original question: {question}
        SQL query executed: {query}
        
        Results summary:
        - {df_summary['row_count']} rows returned
        - Columns: {df_summary['columns']}
        - Sample data: {df_summary['sample_data']}
        
        Provide a comprehensive analysis that includes:
        1. Direct answer to the question
        2. Key insights and trends
        3. Business implications
        4. Recommendations for action
        5. Additional relevant observations
        
        Be conversational, insightful, and actionable. Use specific numbers from the data.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are Datrik, an intelligent food delivery business analyst."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analyzing results: {str(e)}"
    
    def create_visualizations(self, df: pd.DataFrame, question: str) -> list:
        """Create relevant visualizations based on the data"""
        visualizations = []
        
        if df.empty:
            return visualizations
        
        # Detect time series data
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        try:
            # Time series visualization
            if date_columns and numeric_columns and len(df) > 1:
                date_col = date_columns[0]
                df[date_col] = pd.to_datetime(df[date_col])
                
                for num_col in numeric_columns[:2]:  # Limit to 2 charts
                    if num_col != date_col:
                        fig = px.line(df, x=date_col, y=num_col, 
                                    title=f'{num_col.replace("_", " ").title()} Over Time')
                        fig.update_layout(xaxis_title="Date", yaxis_title=num_col.replace("_", " ").title())
                        visualizations.append(fig)
            
            # Bar chart for categorical data with numeric values
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_columns and numeric_columns and len(df) <= 20:  # Limit for readability
                cat_col = categorical_columns[0]
                num_col = numeric_columns[0]
                
                # Group by categorical column and sum numeric column
                chart_data = df.groupby(cat_col)[num_col].sum().reset_index()
                chart_data = chart_data.sort_values(num_col, ascending=False).head(10)
                
                fig = px.bar(chart_data, x=cat_col, y=num_col,
                           title=f'{num_col.replace("_", " ").title()} by {cat_col.replace("_", " ").title()}')
                fig.update_layout(xaxis_title=cat_col.replace("_", " ").title(), 
                                yaxis_title=num_col.replace("_", " ").title())
                visualizations.append(fig)
            
            # Pie chart for distributions
            if len(categorical_columns) > 0 and len(df) <= 10:
                cat_col = categorical_columns[0]
                if len(numeric_columns) > 0:
                    num_col = numeric_columns[0]
                    fig = px.pie(df, names=cat_col, values=num_col,
                               title=f'Distribution of {num_col.replace("_", " ").title()}')
                    visualizations.append(fig)
        
        except Exception as e:
            st.warning(f"Could not create visualization: {str(e)}")
        
        return visualizations
    
    def get_database_stats(self) -> dict:
        """Get basic database statistics"""
        stats = {}
        try:
            conn = self.get_database_connection()
            
            tables = ['users', 'restaurants', 'orders', 'order_items', 'couriers', 'sessions', 'marketing_events', 'reviews']
            
            for table in tables:
                try:
                    result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                    stats[table] = result[0] if result else 0
                except:
                    stats[table] = 0
            
            # Get date range
            try:
                result = conn.execute("SELECT MIN(order_date), MAX(order_date) FROM orders").fetchone()
                if result and result[0] and result[1]:
                    stats['date_range'] = f"{result[0][:10]} to {result[1][:10]}"
                else:
                    stats['date_range'] = "No data"
            except:
                stats['date_range'] = "Unable to determine"
            
            conn.close()
            
        except Exception as e:
            st.error(f"Error getting database stats: {str(e)}")
            
        return stats

def main():
    st.set_page_config(
        page_title="Datrik - AI Food Delivery Analyst",
        page_icon="🍕",
        layout="wide"
    )
    
    st.title("🍕 Datrik - AI Food Delivery Analyst")
    st.markdown("Ask me anything about your food delivery business data!")
    
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        st.error("⚠️ OpenAI API key not found! Please set your OPENAI_API_KEY in a .env file.")
        st.markdown("""
        Create a `.env` file with:
        ```
        OPENAI_API_KEY=your_openai_api_key_here
        ```
        """)
        return
    
    # Initialize Datrik
    datrik = DatrikAnalyst()
    
    # Check if database exists
    if not os.path.exists(datrik.db_path):
        st.error(f"⚠️ Database not found at {datrik.db_path}")
        st.markdown("Please run the data generator first: `python src/data_generator.py`")
        return
    
    # Sidebar with database stats
    with st.sidebar:
        st.header("📊 Database Overview")
        stats = datrik.get_database_stats()
        
        if stats:
            st.metric("📅 Data Range", stats.get('date_range', 'Unknown'))
            st.metric("👥 Users", f"{stats.get('users', 0):,}")
            st.metric("🏪 Restaurants", f"{stats.get('restaurants', 0):,}")
            st.metric("📦 Orders", f"{stats.get('orders', 0):,}")
            st.metric("🛵 Couriers", f"{stats.get('couriers', 0):,}")
            st.metric("📱 Sessions", f"{stats.get('sessions', 0):,}")
            st.metric("⭐ Reviews", f"{stats.get('reviews', 0):,}")
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        st.markdown("""
        - Which restaurant had the highest orders last week?
        - What was the average order size this month?
        - Show me week over week order trends
        - Which cuisine is most popular?
        - What's our customer retention rate?
        - How effective are our marketing campaigns?
        """)
    
    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display visualizations if present
            if "visualizations" in message:
                for fig in message["visualizations"]:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display data if present
            if "data" in message:
                with st.expander("📊 View Raw Data"):
                    st.dataframe(message["data"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your food delivery data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your question..."):
                
                # Convert natural language to SQL
                sql_query = datrik.natural_language_to_sql(prompt)
                
                if sql_query:
                    st.markdown("**Generated SQL Query:**")
                    st.code(sql_query, language='sql')
                    
                    # Execute query
                    with st.spinner("📊 Executing query..."):
                        df = datrik.execute_query(sql_query)
                    
                    if not df.empty:
                        # Analyze results
                        with st.spinner("🧠 Generating insights..."):
                            analysis = datrik.analyze_results(prompt, sql_query, df)
                        
                        st.markdown("**Analysis:**")
                        st.markdown(analysis)
                        
                        # Create visualizations
                        visualizations = datrik.create_visualizations(df, prompt)
                        
                        # Display visualizations
                        for fig in visualizations:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show raw data in expandable section
                        with st.expander("📊 View Raw Data"):
                            st.dataframe(df)
                        
                        # Store message with all components
                        assistant_message = {
                            "role": "assistant",
                            "content": f"**Generated SQL Query:**\n```sql\n{sql_query}\n```\n\n**Analysis:**\n{analysis}",
                            "visualizations": visualizations,
                            "data": df
                        }
                        
                    else:
                        assistant_message = {
                            "role": "assistant",
                            "content": f"**Generated SQL Query:**\n```sql\n{sql_query}\n```\n\n**Result:** No data found for your query."
                        }
                else:
                    assistant_message = {
                        "role": "assistant",
                        "content": "I couldn't generate a SQL query for your question. Please try rephrasing or asking a different question."
                    }
                
                st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main()