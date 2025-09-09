-- Database views for common queries

-- Active signals view
CREATE OR REPLACE VIEW v_active_signals AS
SELECT 
    s.id,
    s.pair,
    s.direction,
    s.entry_price,
    s.stop_loss,
    s.take_profits,
    s.risk_level,
    s.confidence_score,
    cs.channel_name,
    cs.reputation_score,
    s.created_at
FROM signals s
LEFT JOIN channel_statistics cs ON s.source_channel_id = cs.channel_id
WHERE s.status = 'active'
ORDER BY s.created_at DESC;

-- High confidence signals
CREATE OR REPLACE VIEW v_high_confidence_signals AS
SELECT * FROM v_active_signals
WHERE confidence_score >= 75;

-- Channel performance summary
CREATE OR REPLACE VIEW v_channel_performance AS
SELECT 
    cs.channel_id,
    cs.channel_name,
    cs.total_signals,
    cs.successful_signals,
    cs.failed_signals,
    CASE 
        WHEN cs.total_signals > 0 
        THEN ROUND((cs.successful_signals::DECIMAL / cs.total_signals) * 100, 2)
        ELSE 0 
    END as success_rate,
    cs.average_confidence,
    cs.reputation_score,
    cs.last_signal_at
FROM channel_statistics cs
WHERE cs.total_signals > 0
ORDER BY success_rate DESC;

-- Signal performance overview
CREATE OR REPLACE VIEW v_signal_performance AS
SELECT 
    s.id,
    s.pair,
    s.direction,
    s.entry_price,
    sp.actual_exit,
    sp.pnl_percentage,
    sp.pnl_amount,
    sp.hit_stop_loss,
    sp.hit_take_profit,
    sp.duration_hours,
    s.confidence_score,
    s.created_at,
    sp.closed_at
FROM signals s
INNER JOIN signal_performance sp ON s.id = sp.signal_id
ORDER BY sp.closed_at DESC;

-- Daily signal statistics
CREATE OR REPLACE VIEW v_daily_signal_stats AS
SELECT 
    DATE(created_at) as signal_date,
    COUNT(*) as total_signals,
    COUNT(CASE WHEN direction = 'long' THEN 1 END) as long_signals,
    COUNT(CASE WHEN direction = 'short' THEN 1 END) as short_signals,
    AVG(confidence_score) as avg_confidence,
    COUNT(CASE WHEN risk_level = 'low' THEN 1 END) as low_risk,
    COUNT(CASE WHEN risk_level = 'medium' THEN 1 END) as medium_risk,
    COUNT(CASE WHEN risk_level = 'high' THEN 1 END) as high_risk
FROM signals
GROUP BY DATE(created_at)
ORDER BY signal_date DESC;

-- Recent telegram messages requiring processing
CREATE OR REPLACE VIEW v_unprocessed_messages AS
SELECT 
    id,
    channel_id,
    message_id,
    content,
    author,
    timestamp,
    has_media,
    created_at
FROM telegram_messages
WHERE processed = FALSE
ORDER BY timestamp DESC
LIMIT 1000;
