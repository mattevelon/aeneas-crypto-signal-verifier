# Project Memory

## Project Overview
- **Name**: AENEAS - Crypto Trading Signal Verification System
- **Type**: AI-powered cryptocurrency trading signal analysis platform
- **Status**: Phase 1 Infrastructure (95% complete), Phase 2 Data Collection (0%)
- **Tech Stack**: Python 3.11, FastAPI, PostgreSQL, Redis, Qdrant, Kafka, Docker

## Architecture Decisions
- **Async Framework**: FastAPI with full async/await support for high performance
- **Graceful Degradation**: Application can run without non-critical services (Kafka, Telegram, LLM)
- **Service Isolation**: Each service runs in Docker container with health checks
- **Configuration Management**: Pydantic-based settings with environment validation
- **Rate Limiting**: Implemented for Qdrant operations (100 req/min)
- **Caching Strategy**: Redis for market data and signal caching with TTL

## Dependencies
- **Core**: fastapi==0.109.0, uvicorn[standard]==0.27.0, pydantic==2.5.3
- **Database**: asyncpg==0.29.0, sqlalchemy==2.0.25, alembic==1.13.1
- **Messaging**: aiokafka==0.10.0, telethon==1.34.0
- **AI/ML**: openai==1.8.0, langchain==0.1.0, qdrant-client==1.7.0
- **Market Data**: yfinance==0.2.33, ta-lib==0.6.7

## Configuration
- **Environment Files**: .env contains all credentials (Telegram, LLM, Exchange APIs)
- **Docker Services**: PostgreSQL (5432), Redis (6379), Qdrant (6333), Kafka (9092)
- **API Keys**: OpenRouter for LLM, Binance/KuCoin for market data
- **Monitoring**: Prometheus (9090), Grafana (3000), Jaeger (16686) - configured but not running

## Error Handling Patterns
- **Missing Credentials**: Services check for credentials and disable gracefully
- **Service Failures**: Non-critical services (Kafka, Qdrant) won't crash the app
- **Rate Limiting**: Decorator pattern for API rate limiting
- **Transaction Management**: Async context managers with automatic rollback

## Database Schema
- **signals**: Trading signals with entry/exit prices, risk levels
- **telegram_messages**: Raw messages with compression for large content
- **channel_statistics**: Performance tracking per channel
- **Migrations**: Alembic-based with version control
- **Bulk Operations**: Optimized bulk insert/update with batching

## Signal Processing
- **Detection**: Regex patterns + LLM analysis for signal extraction
- **Validation**: Multi-layer validation (market data, risk parameters, liquidity)
- **Risk Assessment**: Automated risk level calculation based on multiple factors
- **Market Integration**: Real-time price validation via Binance API

## Testing Strategy
- **Unit Tests**: Signal detector, validator, configuration
- **Integration Tests**: API endpoints, database operations
- **System Tests**: Full system health check script (test_system.py)
- **Coverage Target**: 80% (configured in pyproject.toml)

## Phase 1 Completion (2024-01-14)
- ✅ Python environment and dependencies
- ✅ Docker infrastructure (all services except monitoring)
- ✅ Database schema and migrations
- ✅ Redis caching layer with persistence
- ✅ Kafka messaging (fixed Docker issues)
- ✅ Qdrant vector DB with rate limiting
- ✅ Core API implementation
- ✅ Signal detection and validation logic
- ✅ Market data integration
- ✅ Async database operations
- ⚠️ Monitoring stack (Docker network issues)

## Known Issues and Solutions
- **Docker Pull Failures**: Network timeout issues with monitoring images
- **Database User**: Test environment needs proper user configuration
- **API Health Endpoint**: Returns 404 (needs /api/v1/health path)
- **Redis Client**: Initialization issue in standalone tests (works in app)

## Manual Tasks Completed
- ❌ Telegram API credentials (provided in .env)
- ❌ LLM API keys (OpenRouter configured)
- ❌ Exchange API credentials (Binance/KuCoin ready)
- ❌ Branch protection rules (requires GitHub access)

## Next Steps (Phase 2)
- Implement Telegram data collector with real channels
- Build signal processing pipeline
- Create WebSocket real-time updates
- Implement performance tracking
- Add comprehensive logging and monitoring
