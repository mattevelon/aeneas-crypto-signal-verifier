# Configuration Guide

## Overview
The Crypto Trading Signal Verification System uses environment variables for configuration. All settings are validated using Pydantic and can be configured via a `.env` file.

## Configuration Files

### `.env`
- **Purpose**: Store actual credentials and environment-specific settings
- **Location**: Project root directory
- **Security**: Never commit to version control (included in `.gitignore`)

### `.env.example`
- **Purpose**: Template showing all available configuration options
- **Location**: Project root directory
- **Usage**: Copy to `.env` and fill in actual values

### `src/config/settings.py`
- **Purpose**: Configuration schema and validation
- **Features**: Type validation, default values, credential checking

## Configuration Categories

### 1. Application Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_ENV` | string | development | Environment (development/staging/production) |
| `APP_PORT` | integer | 8000 | Application server port |
| `LOG_LEVEL` | string | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `DEBUG` | boolean | false | Enable debug mode |

### 2. Database Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATABASE_URL` | string | postgresql://... | PostgreSQL connection URL |
| `DATABASE_POOL_SIZE` | integer | 20 | Connection pool size |
| `DATABASE_MAX_OVERFLOW` | integer | 40 | Maximum overflow connections |
| `DATABASE_POOL_TIMEOUT` | integer | 30 | Pool timeout in seconds |

### 3. Redis Cache

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_URL` | string | redis://localhost:6379/0 | Redis connection URL |
| `REDIS_TTL` | integer | 3600 | Default TTL in seconds |
| `REDIS_MAX_CONNECTIONS` | integer | 50 | Maximum connections |

### 4. Telegram API (Required)

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `TELEGRAM_API_ID` | integer | Yes | Telegram API ID from my.telegram.org |
| `TELEGRAM_API_HASH` | string | Yes | Telegram API hash |
| `TELEGRAM_PHONE_NUMBER` | string | Yes | Phone number (format: +1234567890) |
| `TELEGRAM_SESSION_NAME` | string | No | Session file name |
| `TELEGRAM_CHANNELS` | string | Yes | Comma-separated channel list |

### 5. LLM Configuration (Required)

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `LLM_PROVIDER` | string | Yes | Provider (openai/anthropic) |
| `LLM_MODEL` | string | Yes | Model name |
| `LLM_API_KEY` | string | Yes | API key for LLM provider |
| `LLM_TEMPERATURE` | float | No | Temperature (0.0-1.0) |
| `LLM_MAX_TOKENS` | integer | No | Maximum response tokens |
| `LLM_TIMEOUT` | integer | No | Request timeout in seconds |
| `LLM_MAX_RETRIES` | integer | No | Maximum retry attempts |

### 6. Exchange APIs (Optional)

#### Binance
| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `BINANCE_API_KEY` | string | No | Binance API key |
| `BINANCE_API_SECRET` | string | No | Binance API secret |

#### KuCoin
| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `KUCOIN_API_KEY` | string | No | KuCoin API key |
| `KUCOIN_API_SECRET` | string | No | KuCoin API secret |
| `KUCOIN_API_PASSPHRASE` | string | No | KuCoin API passphrase |

### 7. Vector Database

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VECTOR_DB_URL` | string | http://localhost:6333 | Qdrant URL |
| `QDRANT_API_KEY` | string | None | API key (cloud deployment) |
| `VECTOR_COLLECTION_NAME` | string | signals | Collection name |

### 8. Kafka Message Queue

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `KAFKA_BOOTSTRAP_SERVERS` | string | localhost:9092 | Kafka servers |
| `KAFKA_TOPIC_SIGNALS` | string | crypto-signals | Signals topic |
| `KAFKA_TOPIC_VALIDATION` | string | signal-validation | Validation topic |
| `KAFKA_CONSUMER_GROUP` | string | signal-processor | Consumer group |

### 9. API Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `API_PREFIX` | string | /api/v1 | API route prefix |
| `CORS_ORIGINS` | string | * | Allowed CORS origins |
| `JWT_SECRET_KEY` | string | change-this... | JWT signing key |
| `JWT_ALGORITHM` | string | HS256 | JWT algorithm |
| `JWT_EXPIRATION_MINUTES` | integer | 1440 | Token expiration |

### 10. Rate Limiting

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | integer | 60 | Requests per minute |
| `RATE_LIMIT_BURST_SIZE` | integer | 100 | Burst size |

### 11. Feature Flags

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_WEBSOCKET` | boolean | true | Enable WebSocket support |
| `ENABLE_BACKTESTING` | boolean | true | Enable backtesting |
| `ENABLE_PAPER_TRADING` | boolean | false | Enable paper trading |
| `ENABLE_NOTIFICATIONS` | boolean | true | Enable notifications |

### 12. Performance Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SIGNAL_PROCESSING_TIMEOUT` | integer | 2 | Processing timeout (seconds) |
| `MAX_CONCURRENT_SIGNALS` | integer | 10 | Max concurrent signals |
| `CACHE_TTL_SECONDS` | integer | 3600 | Cache TTL |

### 13. Monitoring

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PROMETHEUS_PORT` | integer | 9090 | Prometheus metrics port |
| `JAEGER_ENDPOINT` | string | http://localhost:14268/api/traces | Jaeger endpoint |
| `JAEGER_SERVICE_NAME` | string | crypto-signals-api | Service name |

## Environment-Specific Settings

### Development
```env
APP_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG
```

### Staging
```env
APP_ENV=staging
DEBUG=false
LOG_LEVEL=INFO
CORS_ORIGINS=https://staging.example.com
```

### Production
```env
APP_ENV=production
DEBUG=false
LOG_LEVEL=WARNING
CORS_ORIGINS=https://api.example.com,https://app.example.com
JWT_SECRET_KEY=<secure-random-string>
```

## Security Best Practices

1. **Never commit `.env` files** to version control
2. **Use strong, unique JWT secret keys** in production
3. **Restrict CORS origins** to specific domains in production
4. **Use read-only API keys** for exchanges when possible
5. **Rotate credentials regularly** (see secret rotation mechanism)
6. **Use environment-specific configurations**
7. **Enable SSL/TLS** in production

## Validation

The application validates all configuration on startup:
- **Critical settings** (DATABASE_URL) will cause startup failure if missing
- **Optional settings** will trigger warnings but allow startup
- **Type validation** ensures correct data types
- **Range validation** for numeric values

## Testing Configuration

Run configuration tests:
```bash
pytest tests/test_config.py -v
```

## Troubleshooting

### Missing Credentials Warning
If you see warnings about missing optional services, check:
1. Telegram credentials (api_id, api_hash, phone_number)
2. LLM API key
3. Exchange API keys

### Database Connection Error
Verify:
1. PostgreSQL is running (`docker-compose up postgres`)
2. DATABASE_URL is correct
3. Database exists and is accessible

### Redis Connection Error
Check:
1. Redis is running (`docker-compose up redis`)
2. REDIS_URL is correct
3. Redis is accessible on specified port
