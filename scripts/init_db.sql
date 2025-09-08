-- Initialize database schema for crypto signals verification system

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Create enum types
CREATE TYPE signal_direction AS ENUM ('long', 'short');
CREATE TYPE risk_level AS ENUM ('low', 'medium', 'high');
CREATE TYPE signal_status AS ENUM ('active', 'closed', 'cancelled', 'expired');

-- Main signals table
CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_channel_id BIGINT NOT NULL,
    original_message_id BIGINT NOT NULL,
    pair VARCHAR(20) NOT NULL,
    direction signal_direction NOT NULL,
    entry_price DECIMAL(18,8) NOT NULL,
    stop_loss DECIMAL(18,8) NOT NULL,
    take_profits JSONB NOT NULL,
    risk_level risk_level,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 100),
    justification JSONB NOT NULL,
    status signal_status DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    CONSTRAINT unique_signal UNIQUE (source_channel_id, original_message_id)
);

-- Create indexes for signals table
CREATE INDEX idx_signals_pair ON signals(pair);
CREATE INDEX idx_signals_status ON signals(status);
CREATE INDEX idx_signals_created_at ON signals(created_at DESC);
CREATE INDEX idx_signals_confidence ON signals(confidence_score DESC);
CREATE INDEX idx_signals_channel ON signals(source_channel_id);
CREATE INDEX idx_signals_metadata ON signals USING gin(metadata);

-- Telegram messages table with partitioning
CREATE TABLE IF NOT EXISTS telegram_messages (
    id BIGSERIAL,
    channel_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL,
    content TEXT,
    author VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    has_media BOOLEAN DEFAULT FALSE,
    media_urls JSONB,
    processed BOOLEAN DEFAULT FALSE,
    is_signal BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create partitions for the last 3 months and next month
CREATE TABLE telegram_messages_2024_10 PARTITION OF telegram_messages
    FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');
CREATE TABLE telegram_messages_2024_11 PARTITION OF telegram_messages
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');
CREATE TABLE telegram_messages_2024_12 PARTITION OF telegram_messages
    FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');
CREATE TABLE telegram_messages_2025_01 PARTITION OF telegram_messages
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Create indexes for telegram_messages
CREATE INDEX idx_telegram_messages_channel ON telegram_messages(channel_id);
CREATE INDEX idx_telegram_messages_processed ON telegram_messages(processed);
CREATE INDEX idx_telegram_messages_is_signal ON telegram_messages(is_signal);

-- Channel statistics table
CREATE TABLE IF NOT EXISTS channel_statistics (
    channel_id BIGINT PRIMARY KEY,
    channel_name VARCHAR(255),
    total_signals INTEGER DEFAULT 0,
    successful_signals INTEGER DEFAULT 0,
    failed_signals INTEGER DEFAULT 0,
    average_confidence FLOAT,
    reputation_score FLOAT,
    last_signal_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Signal performance tracking
CREATE TABLE IF NOT EXISTS signal_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID REFERENCES signals(id) ON DELETE CASCADE,
    actual_entry DECIMAL(18,8),
    actual_exit DECIMAL(18,8),
    pnl_percentage FLOAT,
    pnl_amount DECIMAL(18,8),
    hit_stop_loss BOOLEAN DEFAULT FALSE,
    hit_take_profit INTEGER,
    duration_hours INTEGER,
    closed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for signal performance
CREATE INDEX idx_signal_performance_signal_id ON signal_performance(signal_id);
CREATE INDEX idx_signal_performance_pnl ON signal_performance(pnl_percentage DESC);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for audit log
CREATE INDEX idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX idx_audit_log_entity ON audit_log(entity_type, entity_id);
CREATE INDEX idx_audit_log_created_at ON audit_log(created_at DESC);

-- Create update trigger for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_signals_updated_at BEFORE UPDATE ON signals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_channel_statistics_updated_at BEFORE UPDATE ON channel_statistics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for automatic partition creation
CREATE OR REPLACE FUNCTION create_monthly_partition()
RETURNS void AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    partition_date := DATE_TRUNC('month', NOW() + INTERVAL '1 month');
    partition_name := 'telegram_messages_' || TO_CHAR(partition_date, 'YYYY_MM');
    start_date := partition_date;
    end_date := partition_date + INTERVAL '1 month';
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = partition_name
    ) THEN
        EXECUTE format('CREATE TABLE %I PARTITION OF telegram_messages FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create a scheduled job to create partitions (requires pg_cron extension)
-- SELECT cron.schedule('create-partitions', '0 0 1 * *', 'SELECT create_monthly_partition();');
