# 🍕 Datrik - AI Food Delivery Analyst

Datrik is an intelligent AI analyst for food delivery business data. It allows you to query your business data using natural language and get intelligent insights, trends analysis, and actionable recommendations.

## 🚀 Features

- **Natural Language Queries**: Ask questions in plain English
- **Intelligent SQL Generation**: Converts your questions to optimized SQL queries
- **Smart Analysis**: AI-powered insights and trend analysis
- **Interactive Visualizations**: Auto-generated charts and graphs
- **Realistic Data**: 90 days of synthetic food delivery data with proper seasonality
- **Chat Interface**: User-friendly Streamlit-based chat interface

## 📊 Database Schema

The system includes realistic data for:
- **Users** (15k): Customer demographics and behavior
- **Restaurants** (1.2k): Restaurant details and performance metrics
- **Orders** (~45k): Order history with seasonality patterns
- **Order Items** (~95k): Detailed order contents
- **Couriers** (600): Delivery driver information
- **Sessions** (100k): User app interactions for funnel analysis
- **Marketing Events**: Campaign performance and coupon usage
- **Reviews**: Customer feedback and ratings

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.7 or higher
- OpenAI API key

### Installation

1. **Clone or download the project**
   ```bash
   cd "/Users/tavleen/Projects/datrik 2"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Generate sample data**
   ```bash
   python src/data_generator.py
   ```
   
   This will create:
   - SQLite database at `data/datrik.db`
   - CSV exports in `csv_output/` directory

5. **Launch Datrik**
   ```bash
   streamlit run src/datrik_chat.py
   ```

## 💡 Example Queries

Try asking Datrik questions like:

- "Which restaurant had the highest orders last week?"
- "What was the average order size and how has it changed week over week?"
- "Show me the top 10 most popular menu items"
- "What's our customer retention rate?"
- "Which marketing campaigns are most effective?"
- "How do delivery ratings vary by time of day?"
- "What's the conversion funnel from app opens to orders?"

## 🔧 Configuration

### Data Generation Settings

You can modify data generation parameters in `src/data_generator.py`:

```python
self.config = {
    'users': 15000,           # Number of users
    'restaurants': 1200,      # Number of restaurants
    'couriers': 600,          # Number of couriers
    'avg_orders_per_day': 500, # Average orders per day
    'avg_items_per_order': 2.1, # Average items per order
}
```

### Database Configuration

- **Database Path**: Set in `.env` file (`DATABASE_PATH=./data/datrik.db`)
- **Random Seed**: Set in `.env` file (`RANDOM_SEED=42`)
- **Days of Data**: Set in `.env` file (`DAYS_OF_DATA=90`)

## 📁 Project Structure

```
datrik/
├── src/
│   ├── database_schema.sql    # SQLite database schema
│   ├── data_generator.py      # Realistic data generation
│   └── datrik_chat.py        # Streamlit chat interface
├── data/
│   └── datrik.db             # SQLite database (generated)
├── csv_output/               # CSV exports (generated)
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── README.md               # This file
```

## 🤖 How It Works

1. **Natural Language Processing**: Your question is sent to OpenAI GPT-4
2. **SQL Generation**: AI converts your question to optimized SQL queries
3. **Data Retrieval**: Query is executed against the SQLite database
4. **Intelligent Analysis**: AI analyzes results and provides insights
5. **Visualization**: Auto-generated charts based on data patterns
6. **Interactive Display**: Results shown in user-friendly chat interface

## 🔍 Technical Details

### Data Generation
- **Deterministic**: Uses fixed seed (42) for reproducible results
- **Realistic Patterns**: Includes day-of-week and hour-of-day seasonality
- **Related Data**: Maintains proper relationships between entities
- **Business Logic**: Realistic pricing, delivery times, and customer behavior

### AI Integration
- **Model**: OpenAI GPT-4 for SQL generation and analysis
- **Context Aware**: Includes full database schema in prompts
- **Error Handling**: Graceful handling of invalid queries or API errors

### Performance
- **Indexed Database**: Optimized SQLite indexes for common queries
- **Efficient Queries**: AI generates optimized SQL with proper JOINs and filters
- **Caching**: Streamlit built-in caching for better performance

## 🚨 Troubleshooting

### Common Issues

1. **"Database not found"**
   - Run `python src/data_generator.py` to generate the database

2. **"OpenAI API key not found"**
   - Create `.env` file with your OpenAI API key
   - Make sure the key starts with `sk-`

3. **"Module not found" errors**
   - Install dependencies: `pip install -r requirements.txt`

4. **Slow performance**
   - Check your internet connection (needed for OpenAI API)
   - Try simpler queries first

### Support

For issues or questions:
1. Check the error message in the Streamlit interface
2. Verify your OpenAI API key is valid and has credits
3. Ensure the database was generated successfully

## 🎯 Future Enhancements

- Support for multiple database backends
- Custom visualization templates
- Export capabilities for reports
- Real-time data ingestion
- Advanced analytics and ML predictions
- Multi-language support
- Dashboard templates for common business metrics

---

**Built with ❤️ for food delivery analytics**