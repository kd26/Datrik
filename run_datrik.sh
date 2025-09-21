#!/bin/bash

echo "🍕 Starting Datrik AI Analyst..."
echo "Press Ctrl+C to stop"
echo "================================"

cd "/Users/tavleen/Projects/datrik 2"

# Kill any existing streamlit processes
pkill -f streamlit

# Wait a moment
sleep 2

# Start streamlit
python3 -m streamlit run src/datrik_chat.py --server.port 8501 --server.headless true