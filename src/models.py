"""
Database models for the crypto signals verification system.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
from enum import Enum
from uuid import uuid4

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    ForeignKey, JSON, DECIMAL, BigInteger, Text, 
    Enum as SQLEnum, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SignalStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class Signal(Base):
    """Main signals table."""
    __tablename__ = "signals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_channel_id = Column(BigInteger, nullable=False)
    original_message_id = Column(BigInteger, nullable=False)
    pair = Column(String(20), nullable=False)
    direction = Column(SQLEnum(SignalDirection), nullable=False)
    entry_price = Column(DECIMAL(18, 8), nullable=False)
    stop_loss = Column(DECIMAL(18, 8), nullable=False)
    take_profits = Column(JSONB, nullable=False)
    risk_level = Column(SQLEnum(RiskLevel))
    confidence_score = Column(Float)
    justification = Column(JSONB, nullable=False)
    status = Column(SQLEnum(SignalStatus), default=SignalStatus.ACTIVE)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSONB)
    
    # Relationships
    performance = relationship("SignalPerformance", back_populates="signal", uselist=False)
    
    __table_args__ = (
        UniqueConstraint('source_channel_id', 'original_message_id', name='unique_signal'),
        Index('idx_signals_pair', 'pair'),
        Index('idx_signals_status', 'status'),
        Index('idx_signals_created_at', 'created_at'),
        Index('idx_signals_confidence', 'confidence_score'),
        Index('idx_signals_channel', 'source_channel_id'),
    )


class TelegramMessage(Base):
    """Telegram messages table."""
    __tablename__ = "telegram_messages"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    channel_id = Column(BigInteger, nullable=False)
    message_id = Column(BigInteger, nullable=False)
    content = Column(Text)
    author = Column(String(255))
    timestamp = Column(DateTime(timezone=True), nullable=False)
    has_media = Column(Boolean, default=False)
    media_urls = Column(JSONB)
    processed = Column(Boolean, default=False)
    is_signal = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_telegram_messages_channel', 'channel_id'),
        Index('idx_telegram_messages_processed', 'processed'),
        Index('idx_telegram_messages_is_signal', 'is_signal'),
    )


class ChannelStatistics(Base):
    """Channel statistics table."""
    __tablename__ = "channel_statistics"
    
    channel_id = Column(BigInteger, primary_key=True)
    channel_name = Column(String(255))
    total_signals = Column(Integer, default=0)
    successful_signals = Column(Integer, default=0)
    failed_signals = Column(Integer, default=0)
    average_confidence = Column(Float)
    reputation_score = Column(Float)
    last_signal_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class SignalPerformance(Base):
    """Signal performance tracking table."""
    __tablename__ = "signal_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    signal_id = Column(UUID(as_uuid=True), ForeignKey('signals.id', ondelete='CASCADE'))
    actual_entry = Column(DECIMAL(18, 8))
    actual_exit = Column(DECIMAL(18, 8))
    pnl_percentage = Column(Float)
    pnl_amount = Column(DECIMAL(18, 8))
    hit_stop_loss = Column(Boolean, default=False)
    hit_take_profit = Column(Integer)
    duration_hours = Column(Integer)
    closed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    signal = relationship("Signal", back_populates="performance")
    
    __table_args__ = (
        Index('idx_signal_performance_signal_id', 'signal_id'),
        Index('idx_signal_performance_pnl', 'pnl_percentage'),
    )


class AuditLog(Base):
    """Audit log table."""
    __tablename__ = "audit_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255))
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(String(255))
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    ip_address = Column(String(45))  # Support IPv6
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_audit_log_user_id', 'user_id'),
        Index('idx_audit_log_entity', 'entity_type', 'entity_id'),
        Index('idx_audit_log_created_at', 'created_at'),
    )
