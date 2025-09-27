-- LangChain Memory System Database Schema
-- Add these tables to support conversation memory

-- Conversation sessions for grouping related conversations
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id TEXT PRIMARY KEY,
    user_identifier TEXT,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conversation_summary TEXT,
    total_messages INTEGER DEFAULT 0
);

-- Individual memory entries for each conversation turn
CREATE TABLE IF NOT EXISTS conversation_memory (
    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    message_type TEXT CHECK (message_type IN ('human', 'ai', 'system')),
    content TEXT NOT NULL,
    metadata TEXT, -- JSON string for additional context
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    token_count INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
);

-- User preferences learned from conversations
CREATE TABLE IF NOT EXISTS user_preferences (
    pref_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_identifier TEXT,
    preference_type TEXT, -- 'cuisine', 'time_range', 'analysis_type', etc.
    preference_value TEXT,
    confidence_score REAL DEFAULT 0.5,
    learned_from_session TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_session_activity ON conversation_sessions(last_activity);
CREATE INDEX IF NOT EXISTS idx_memory_session ON conversation_memory(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON conversation_memory(timestamp);
CREATE INDEX IF NOT EXISTS idx_preferences_user ON user_preferences(user_identifier);