# Crypto Trading Signal Verification System - Technical Documentation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Technology Stack](#technology-stack)
5. [Data Flow](#data-flow)
6. [API Specifications](#api-specifications)
7. [Security Architecture](#security-architecture)
8. [Performance Specifications](#performance-specifications)
9. [Deployment Architecture](#deployment-architecture)
10. [Monitoring and Observability](#monitoring-and-observability)

---

## Executive Summary

### Project Overview
The Crypto Trading Signal Verification System is an AI-powered platform designed to verify and analyze cryptocurrency trading signals from Telegram channels. The system employs a single Large Language Model (LLM) accessed via API to provide deep analysis, risk assessment, and comprehensive justification for each signal.

### Key Features
- **Real-time Signal Detection**: Automated extraction of trading signals from Telegram channels
- **Deep AI Analysis**: Comprehensive verification using advanced LLM capabilities
- **Multi-level Validation**: Technical, fundamental, and market context validation
- **Risk Management**: Sophisticated risk assessment and position sizing
- **Detailed Justification**: Multi-tiered explanations for traders of all levels
- **High Performance**: Sub-2-second signal processing with 99.9% uptime

### Business Goals
- Provide reliable trading signal verification for cryptocurrency traders
- Reduce false signals through AI-powered analysis
- Offer transparent, educational insights into trading decisions
- Support both novice and expert traders with appropriate detail levels

---

## System Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph External Sources
        TG[Telegram Channels]
        MA[Market APIs]
        OCR[OCR Services]
    end
    
    subgraph Data Ingestion Layer
        TC[Telegram Collector]
        IP[Image Processor]
        DP[Data Preprocessor]
    end
    
    subgraph Storage Layer
        PG[(PostgreSQL)]
        VDB[(Vector DB)]
        RD[(Redis Cache)]
    end
    
    subgraph Processing Core
        SD[Signal Detector]
        CM[Context Manager]
        PM[Prompt Manager]
        LLM[LLM API Client]
    end
    
    subgraph Validation Layer
        MV[Market Validator]
        RA[Risk Analyzer]
        JG[Justification Generator]
    end
    
    subgraph API Layer
        REST[REST API]
        WS[WebSocket Server]
        AUTH[Auth Service]
    end
    
    subgraph Client Layer
        WEB[Web Dashboard]
        TMA[Telegram Mini App]
        MOB[Mobile App]
    end
    
    TG --> TC
    TC --> DP
    IP --> DP
    DP --> SD
    
    SD --> CM
    CM --> PM
    PM --> LLM
    LLM --> MV
    MV --> RA
    RA --> JG
    
    DP --> PG
    DP --> VDB
    CM --> VDB
    CM --> RD
    
    MA --> MV
    OCR --> IP
    
    JG --> REST
    JG --> WS
    
    REST --> WEB
    WS --> TMA
    REST --> MOB
    
    AUTH --> REST
    AUTH --> WS
```

### Component Architecture

The system follows a microservices-inspired modular architecture with clear separation of concerns:

1. **Data Ingestion Layer**: Handles all external data collection
2. **Storage Layer**: Manages persistent and cached data
3. **Processing Core**: Contains business logic and AI integration
4. **Validation Layer**: Ensures signal quality and generates outputs
5. **API Layer**: Provides external interfaces
6. **Client Layer**: User-facing applications

---

## Core Components

### 1. Telegram Collector
**Purpose**: Real-time data ingestion from Telegram channels

**Technical Specifications**:
- Framework: Telethon (MTProto protocol)
- Concurrency: AsyncIO-based event loop
- Connection Pool: 10 persistent connections
- Rate Limiting: Exponential backoff (base: 1s, max: 32s)
- Message Queue: Priority queue with 10,000 message capacity

**Key Features**:
- Multi-channel simultaneous monitoring
- Automatic reconnection on disconnect
- Message deduplication
- Media download handling
- Channel health monitoring

### 2. Signal Detector
**Purpose**: Identify and extract trading signals from messages

**Technical Specifications**:
- Pattern Matching: Regex engine with 50+ patterns
- ML Classification: BERT-based classifier (F1 score: 0.92)
- Confidence Scoring: Ensemble voting mechanism
- Processing Time: <100ms per message

**Signal Parameters Extracted**:
- Trading pair (e.g., BTC/USDT)
- Entry price
- Stop-loss levels
- Take-profit targets (multiple)
- Position type (long/short)
- Timeframe
- Leverage (if applicable)

### 3. Context Manager
**Purpose**: Aggregate relevant context for signal analysis

**Technical Specifications**:
- Time Window: 24-hour sliding window
- Vector Search: k-NN with k=20
- Similarity Threshold: Cosine similarity > 0.85
- Cache TTL: 3600 seconds
- Token Budget: 8000 tokens max

**Context Components**:
- Historical similar signals
- Market data snapshot
- Technical indicators (RSI, MACD, Bollinger Bands)
- Channel reputation metrics
- Cross-channel validation data
- News sentiment analysis

### 4. LLM Integration Client
**Purpose**: Interface with AI model for analysis

**Technical Specifications**:
- Model: GPT-4-turbo / Claude-3-opus
- Temperature: 0.3 for consistency
- Max Tokens: 4000 per request
- Timeout: 30 seconds
- Retry Strategy: 3 attempts with exponential backoff

**Optimization Features**:
- Request batching (up to 5 signals)
- Response caching (1-hour TTL)
- Token usage tracking
- Cost monitoring
- Fallback model switching

### 5. Validation Engine
**Purpose**: Multi-level signal validation

**Validation Layers**:
1. **Market Data Validation**
   - Price deviation check (< 2% from current)
   - Volume verification (> $100k daily)
   - Spread analysis (< 0.5%)
   
2. **Risk Assessment**
   - Position sizing (Kelly Criterion)
   - Risk/Reward ratio (minimum 1:1.5)
   - Maximum drawdown calculation
   
3. **Manipulation Detection**
   - Pump & dump patterns
   - Wash trading indicators
   - Unusual volume spikes

### 6. Justification Generator
**Purpose**: Create comprehensive explanations

**Output Tiers**:
1. **Novice** (2-3 sentences): Basic signal explanation
2. **Intermediate** (1-2 paragraphs): Technical context
3. **Expert** (full analysis): Complete breakdown with metrics

**Languages Supported**: English, Russian, Chinese, Spanish

---

## Technology Stack

### Backend Technologies
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Runtime | Python | 3.11+ | Core application runtime |
| Framework | FastAPI | 0.109.0 | REST API framework |
| Async | AsyncIO | Native | Asynchronous processing |
| Database | PostgreSQL | 15+ | Primary data storage |
| Vector DB | Qdrant | 1.7.0 | Semantic search |
| Cache | Redis | 7.0 | In-memory caching |
| Message Queue | Kafka | 3.5 | Event streaming |

### AI/ML Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | OpenAI GPT-4 / Anthropic Claude | Signal analysis |
| Embeddings | text-embedding-3-small | Vector generation |
| OCR | Google Vision API | Chart text extraction |
| NLP | spaCy | Text processing |
| Technical Analysis | TA-Lib | Indicator calculation |

### Infrastructure
| Component | Technology | Purpose |
|-----------|------------|---------|
| Containerization | Docker | Application packaging |
| Orchestration | Kubernetes | Container management |
| CI/CD | GitHub Actions | Automated deployment |
| Monitoring | Prometheus + Grafana | Metrics and visualization |
| Logging | ELK Stack | Log aggregation |
| Tracing | OpenTelemetry | Distributed tracing |

---

## Data Flow

### Signal Processing Pipeline

```mermaid
sequenceDiagram
    participant TG as Telegram
    participant TC as Collector
    participant SD as Signal Detector
    participant CM as Context Manager
    participant AI as AI Model
    participant VAL as Validator
    participant API as API
    participant USER as User
    
    TG->>TC: New Message
    TC->>SD: Process Message
    SD->>SD: Detect Signal
    
    alt Signal Detected
        SD->>CM: Build Context
        CM->>CM: Gather Data
        CM->>AI: Analyze Signal
        AI->>AI: Deep Analysis
        AI->>VAL: Analysis Result
        VAL->>VAL: Validate
        
        alt Validation Passed
            VAL->>API: Publish Signal
            API->>USER: Notify
        else Validation Failed
            VAL->>VAL: Log Rejection
        end
    else No Signal
        SD->>SD: Archive Message
    end
```

### Data Storage Schema

#### PostgreSQL Tables

```sql
-- Main signals table
CREATE TABLE signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_channel_id BIGINT NOT NULL,
    original_message_id BIGINT NOT NULL,
    pair VARCHAR(20) NOT NULL,
    direction VARCHAR(10) CHECK (direction IN ('long', 'short')),
    entry_price DECIMAL(18,8) NOT NULL,
    stop_loss DECIMAL(18,8) NOT NULL,
    take_profits JSONB NOT NULL,
    risk_level VARCHAR(20) CHECK (risk_level IN ('low', 'medium', 'high')),
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 100),
    justification JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Channel statistics
CREATE TABLE channel_statistics (
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

-- Performance tracking
CREATE TABLE signal_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID REFERENCES signals(id),
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
```

#### Vector Database Schema

```python
collection_schema = {
    "name": "signals",
    "vectors": {
        "size": 1536,  # OpenAI embedding dimension
        "distance": "Cosine"
    },
    "payload_schema": {
        "signal_id": "keyword",
        "channel_id": "integer",
        "timestamp": "datetime",
        "pair": "keyword",
        "confidence": "float"
    }
}
```

---

## API Specifications

### REST API Endpoints

#### Authentication
```http
POST /api/v1/auth/login
POST /api/v1/auth/refresh
POST /api/v1/auth/logout
```

#### Signals
```http
GET /api/v1/signals                 # List signals with pagination
GET /api/v1/signals/{signal_id}     # Get signal details
GET /api/v1/signals/active          # Get active signals
POST /api/v1/signals/feedback       # Submit signal feedback
```

#### Statistics
```http
GET /api/v1/stats/overview          # System statistics
GET /api/v1/stats/channels          # Channel performance
GET /api/v1/stats/performance       # Signal performance metrics
```

### WebSocket Events

#### Client -> Server
```javascript
{
  "type": "subscribe",
  "channels": ["signals", "alerts"],
  "filters": {
    "pairs": ["BTC/USDT", "ETH/USDT"],
    "risk_level": ["low", "medium"],
    "min_confidence": 70
  }
}
```

#### Server -> Client
```javascript
{
  "type": "new_signal",
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "pair": "BTC/USDT",
    "direction": "long",
    "entry_price": 45000.00,
    "confidence_score": 85.5,
    "risk_level": "medium",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Rate Limiting
- Authentication endpoints: 5 requests/minute
- Public endpoints: 100 requests/minute
- WebSocket connections: 5 per user
- Burst allowance: 150% for 10 seconds

---

## Security Architecture

### Authentication & Authorization
- **Method**: JWT with refresh tokens
- **Token Lifetime**: Access (15 min), Refresh (7 days)
- **MFA Support**: TOTP-based 2FA
- **Role-Based Access**: Admin, Analyst, User, Guest

### Data Security
- **Encryption at Rest**: AES-256-GCM
- **Encryption in Transit**: TLS 1.3
- **Database Encryption**: Transparent Data Encryption (TDE)
- **Key Management**: AWS KMS / HashiCorp Vault

### API Security
- **Rate Limiting**: Token bucket algorithm
- **DDoS Protection**: CloudFlare integration
- **Input Validation**: Pydantic models
- **SQL Injection Prevention**: Parameterized queries
- **XSS Protection**: Content Security Policy headers

### Compliance
- **GDPR**: User data erasure, consent management
- **Data Retention**: 90 days for messages, 1 year for signals
- **Audit Logging**: All operations logged with user context
- **PII Handling**: Anonymization and pseudonymization

---

## Performance Specifications

### System Requirements
| Metric | Requirement | Current Performance |
|--------|-------------|-------------------|
| Signal Processing Latency | < 2 seconds | 1.3 seconds (p50) |
| API Response Time | < 200ms (p95) | 150ms (p95) |
| System Availability | 99.9% | 99.95% |
| Concurrent Users | 10,000 | Tested: 15,000 |
| Signals/Day | 5,000 | Capacity: 10,000 |
| WebSocket Connections | 5,000 | Tested: 7,500 |

### Resource Requirements
| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| API Servers (x3) | 4 vCPU | 8 GB | 50 GB SSD | 1 Gbps |
| Processing Workers (x5) | 8 vCPU | 16 GB | 100 GB SSD | 1 Gbps |
| PostgreSQL Primary | 16 vCPU | 32 GB | 500 GB SSD | 10 Gbps |
| PostgreSQL Replica | 8 vCPU | 16 GB | 500 GB SSD | 10 Gbps |
| Redis Cluster | 4 vCPU | 16 GB | 50 GB SSD | 10 Gbps |
| Vector DB | 8 vCPU | 32 GB | 200 GB NVMe | 10 Gbps |

### Scalability
- **Horizontal Scaling**: API servers and workers
- **Vertical Scaling**: Database and cache layers
- **Auto-scaling Triggers**: CPU > 70%, Memory > 80%
- **Load Balancing**: Round-robin with health checks

---

## Deployment Architecture

### Production Environment

```mermaid
graph TB
    subgraph Internet
        CF[CloudFlare CDN]
        Users[Users]
    end
    
    subgraph AWS/GCP
        subgraph Public Subnet
            ALB[Application Load Balancer]
            NAT[NAT Gateway]
        end
        
        subgraph Private Subnet - AZ1
            API1[API Server 1]
            WORK1[Worker 1]
            WORK2[Worker 2]
        end
        
        subgraph Private Subnet - AZ2
            API2[API Server 2]
            WORK3[Worker 3]
            WORK4[Worker 4]
        end
        
        subgraph Database Subnet
            PG_PRIMARY[(PostgreSQL Primary)]
            PG_REPLICA[(PostgreSQL Replica)]
            REDIS[(Redis Cluster)]
            VECTOR[(Vector DB)]
        end
    end
    
    Users --> CF
    CF --> ALB
    ALB --> API1
    ALB --> API2
    API1 --> PG_PRIMARY
    API2 --> PG_REPLICA
    WORK1 --> REDIS
    WORK2 --> VECTOR
```

### Container Orchestration

```yaml
# Kubernetes Deployment Example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: signal-processor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: signal-processor
  template:
    metadata:
      labels:
        app: signal-processor
    spec:
      containers:
      - name: processor
        image: crypto-signals/processor:v1.0.0
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: llm-key
```

### CI/CD Pipeline

```mermaid
graph LR
    DEV[Developer] --> GIT[Git Push]
    GIT --> CI[GitHub Actions]
    CI --> TEST[Run Tests]
    TEST --> BUILD[Build Docker Image]
    BUILD --> SCAN[Security Scan]
    SCAN --> PUSH[Push to Registry]
    PUSH --> DEPLOY_STG[Deploy to Staging]
    DEPLOY_STG --> E2E[E2E Tests]
    E2E --> APPROVE[Manual Approval]
    APPROVE --> DEPLOY_PROD[Deploy to Production]
    DEPLOY_PROD --> MONITOR[Monitor]
```

---

## Monitoring and Observability

### Metrics Collection
**Tool**: Prometheus + Grafana

**Key Metrics**:
- Signal processing rate
- API latency percentiles (p50, p95, p99)
- LLM API usage and costs
- Database query performance
- Cache hit rates
- Error rates by component

### Logging Strategy
**Tool**: ELK Stack (Elasticsearch, Logstash, Kibana)

**Log Levels**:
- ERROR: System errors requiring immediate attention
- WARN: Potential issues or degraded performance
- INFO: General operational information
- DEBUG: Detailed debugging information

**Log Format**:
```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "service": "signal-processor",
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "message": "Signal processed successfully",
  "metadata": {
    "signal_id": "sig_456",
    "processing_time_ms": 1234,
    "confidence_score": 85.5
  }
}
```

### Alerting Rules

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| High Error Rate | Error rate > 1% for 5 min | Critical | Page on-call |
| API Latency | p95 > 500ms for 10 min | Warning | Slack notification |
| Low Signal Confidence | Avg confidence < 60% for 1 hour | Warning | Email team |
| Database Connection Pool | Connections > 80% | Warning | Auto-scale |
| LLM API Failures | Failure rate > 5% | Critical | Switch to backup |

### Distributed Tracing
**Tool**: OpenTelemetry + Jaeger

**Trace Points**:
- API request entry
- Database queries
- LLM API calls
- Cache operations
- External API calls
- Response generation

---

## Appendix

### Glossary
- **LLM**: Large Language Model
- **RAG**: Retrieval-Augmented Generation
- **k-NN**: k-Nearest Neighbors
- **TTL**: Time To Live
- **SLA**: Service Level Agreement
- **p95/p99**: 95th/99th percentile
- **vCPU**: Virtual CPU
- **AZ**: Availability Zone

### Environment Variables
```bash
# Application
APP_ENV=production
APP_PORT=8000
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/signals
DATABASE_POOL_SIZE=20

# Redis
REDIS_URL=redis://localhost:6379
REDIS_TTL=3600

# LLM API
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
LLM_API_KEY=sk-...
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=4000

# Telegram
TELEGRAM_API_ID=123456
TELEGRAM_API_HASH=abc...
TELEGRAM_CHANNELS=channel1,channel2

# Vector Database
VECTOR_DB_URL=http://localhost:6333
VECTOR_COLLECTION=signals

# Monitoring
PROMETHEUS_PORT=9090
JAEGER_ENDPOINT=http://localhost:14268
```

### Error Codes
| Code | Description | HTTP Status |
|------|-------------|------------|
| SIG001 | Invalid signal format | 400 |
| SIG002 | Signal validation failed | 422 |
| AUTH001 | Invalid credentials | 401 |
| AUTH002 | Token expired | 401 |
| AUTH003 | Insufficient permissions | 403 |
| API001 | Rate limit exceeded | 429 |
| API002 | Service unavailable | 503 |
| LLM001 | AI analysis failed | 500 |
| LLM002 | Token limit exceeded | 400 |
| DB001 | Database connection error | 500 |

---

## Version History
- **v1.0.0** (2025-01-16): Initial release - Full production system with all 6 phases complete
  - Complete signal detection and validation pipeline
  - Multi-provider LLM integration (GPT-4, Claude, DeepSeek)
  - Market validation and risk assessment
  - ML pipeline with XGBoost/LightGBM
  - JWT authentication and WebSocket support
  - Comprehensive monitoring stack
- **v1.1.0** (2025-09-18): DeepSeek V3.1 Integration
  - Integrated DeepSeek V3.1 via OpenRouter API
  - Custom 6,410 character system prompt
  - Optimized for crypto signal analysis
  - 68% confidence score achieved in testing
- **v1.2.0** (Planned): Enhanced ML models and backtesting
- **v2.0.0** (Future): Multi-model ensemble with federated learning

---

## Contact Information
- **Project Owner**: AENEAS Development Team
- **Repository**: mattevelon/aeneas-crypto-signal-verifier
- **Technical Support**: [Contact via repository issues]
- **Security Reports**: [Submit via private repository security advisory]
- **Documentation**: [Available in /docs directory]
