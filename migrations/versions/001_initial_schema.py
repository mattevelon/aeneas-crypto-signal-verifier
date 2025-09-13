"""Initial schema with signals and telegram tables

Revision ID: 001
Revises: 
Create Date: 2025-01-13 15:02:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum types
    op.execute("CREATE TYPE signal_direction AS ENUM ('long', 'short')")
    op.execute("CREATE TYPE risk_level AS ENUM ('low', 'medium', 'high')")
    op.execute("CREATE TYPE signal_status AS ENUM ('active', 'closed', 'cancelled', 'expired')")
    
    # Create signals table
    op.create_table('signals',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source_channel_id', sa.BigInteger(), nullable=False),
        sa.Column('original_message_id', sa.BigInteger(), nullable=False),
        sa.Column('pair', sa.String(length=20), nullable=False),
        sa.Column('direction', postgresql.ENUM('long', 'short', name='signal_direction'), nullable=False),
        sa.Column('entry_price', sa.DECIMAL(precision=18, scale=8), nullable=False),
        sa.Column('stop_loss', sa.DECIMAL(precision=18, scale=8), nullable=False),
        sa.Column('take_profits', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('risk_level', postgresql.ENUM('low', 'medium', 'high', name='risk_level'), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('justification', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', postgresql.ENUM('active', 'closed', 'cancelled', 'expired', name='signal_status'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_channel_id', 'original_message_id', name='unique_signal')
    )
    op.create_index('idx_signals_channel', 'signals', ['source_channel_id'], unique=False)
    op.create_index('idx_signals_confidence', 'signals', ['confidence_score'], unique=False)
    op.create_index('idx_signals_created_at', 'signals', ['created_at'], unique=False)
    op.create_index('idx_signals_pair', 'signals', ['pair'], unique=False)
    op.create_index('idx_signals_status', 'signals', ['status'], unique=False)
    
    # Create telegram_messages table
    op.create_table('telegram_messages',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('channel_id', sa.BigInteger(), nullable=False),
        sa.Column('message_id', sa.BigInteger(), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('author', sa.String(length=255), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('has_media', sa.Boolean(), nullable=True),
        sa.Column('media_urls', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('processed', sa.Boolean(), nullable=True),
        sa.Column('is_signal', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_telegram_messages_channel', 'telegram_messages', ['channel_id'], unique=False)
    op.create_index('idx_telegram_messages_is_signal', 'telegram_messages', ['is_signal'], unique=False)
    op.create_index('idx_telegram_messages_processed', 'telegram_messages', ['processed'], unique=False)
    
    # Create channel_statistics table
    op.create_table('channel_statistics',
        sa.Column('channel_id', sa.BigInteger(), nullable=False),
        sa.Column('channel_name', sa.String(length=255), nullable=True),
        sa.Column('total_signals', sa.Integer(), nullable=True),
        sa.Column('successful_signals', sa.Integer(), nullable=True),
        sa.Column('failed_signals', sa.Integer(), nullable=True),
        sa.Column('average_confidence', sa.Float(), nullable=True),
        sa.Column('reputation_score', sa.Float(), nullable=True),
        sa.Column('last_signal_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('channel_id')
    )
    
    # Create signal_performance table
    op.create_table('signal_performance',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('signal_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('actual_entry', sa.DECIMAL(precision=18, scale=8), nullable=True),
        sa.Column('actual_exit', sa.DECIMAL(precision=18, scale=8), nullable=True),
        sa.Column('pnl_percentage', sa.Float(), nullable=True),
        sa.Column('pnl_amount', sa.DECIMAL(precision=18, scale=8), nullable=True),
        sa.Column('hit_stop_loss', sa.Boolean(), nullable=True),
        sa.Column('hit_take_profit', sa.Integer(), nullable=True),
        sa.Column('duration_hours', sa.Integer(), nullable=True),
        sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['signal_id'], ['signals.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_signal_performance_pnl', 'signal_performance', ['pnl_percentage'], unique=False)
    op.create_index('idx_signal_performance_signal_id', 'signal_performance', ['signal_id'], unique=False)
    
    # Create audit_log table
    op.create_table('audit_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('entity_type', sa.String(length=50), nullable=True),
        sa.Column('entity_id', sa.String(length=255), nullable=True),
        sa.Column('old_values', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('new_values', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_audit_log_created_at', 'audit_log', ['created_at'], unique=False)
    op.create_index('idx_audit_log_entity', 'audit_log', ['entity_type', 'entity_id'], unique=False)
    op.create_index('idx_audit_log_user_id', 'audit_log', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_audit_log_user_id', table_name='audit_log')
    op.drop_index('idx_audit_log_entity', table_name='audit_log')
    op.drop_index('idx_audit_log_created_at', table_name='audit_log')
    op.drop_table('audit_log')
    op.drop_index('idx_signal_performance_signal_id', table_name='signal_performance')
    op.drop_index('idx_signal_performance_pnl', table_name='signal_performance')
    op.drop_table('signal_performance')
    op.drop_table('channel_statistics')
    op.drop_index('idx_telegram_messages_processed', table_name='telegram_messages')
    op.drop_index('idx_telegram_messages_is_signal', table_name='telegram_messages')
    op.drop_index('idx_telegram_messages_channel', table_name='telegram_messages')
    op.drop_table('telegram_messages')
    op.drop_index('idx_signals_status', table_name='signals')
    op.drop_index('idx_signals_pair', table_name='signals')
    op.drop_index('idx_signals_created_at', table_name='signals')
    op.drop_index('idx_signals_confidence', table_name='signals')
    op.drop_index('idx_signals_channel', table_name='signals')
    op.drop_table('signals')
    
    # Drop enum types
    op.execute("DROP TYPE signal_status")
    op.execute("DROP TYPE risk_level")
    op.execute("DROP TYPE signal_direction")
