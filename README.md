# Crypto Trading Signal Verification System

An AI-powered platform for verifying cryptocurrency trading signals from Telegram channels with deep analysis capabilities.

## 🚀 Features

- **Real-time Signal Detection**: Automated extraction of trading signals from Telegram channels
- **Deep AI Analysis**: Comprehensive verification using GPT-4-turbo/Claude-3-opus
- **Multi-level Validation**: Technical, fundamental, and market context validation
- **Risk Management**: Sophisticated risk assessment and position sizing
- **Detailed Justification**: Multi-tiered explanations for traders of all levels
- **High Performance**: Sub-2-second signal processing with 99.9% uptime

## 📋 Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7.0+
- API Keys for:
  - Telegram (API ID and Hash)
  - OpenAI or Anthropic
  - Binance/KuCoin (for market data)

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd "aeneas_architecture of AI work"
```

### 2. Set up Python environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### 3. Configure environment variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys and configuration
nano .env
```

### 4. Start infrastructure services

```bash
# Start all services with Docker Compose
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 5. Initialize the database

```bash
# The database will be automatically initialized with the schema
# when PostgreSQL starts (via init_db.sql)
```

### 6. Install pre-commit hooks (for development)

```bash
pre-commit install
```

## 🏗️ Project Structure

```
.
├── src/                    # Source code
│   ├── api/               # FastAPI application and endpoints
│   ├── core/              # Core business logic
│   ├── data_ingestion/    # Telegram and data collection
│   ├── storage/           # Database and cache layers
│   ├── validation/        # Signal validation logic
│   ├── utils/             # Utility functions
│   └── config/            # Configuration management
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── scripts/               # Utility scripts
├── config/                # Configuration files
│   ├── prometheus/        # Prometheus configuration
│   └── grafana/           # Grafana dashboards
├── docker-compose.yml     # Docker services configuration
├── Dockerfile             # Application container
├── requirements.txt       # Python dependencies
└── .env.example          # Environment variables template
```

## 🚦 Running the Application

### Development Mode

```bash
# Activate virtual environment
source venv/bin/activate

# Run the FastAPI application
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# Build and run with Docker
docker-compose up --build app
```

## 📊 Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Jaeger**: http://localhost:16686
- **API Documentation**: http://localhost:8000/docs

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
```

## 🔧 Development Tools

### Code Formatting

```bash
# Format code with Black
black src/ tests/

# Sort imports
isort src/ tests/
```

### Linting

```bash
# Run flake8
flake8 src/ tests/

# Run mypy for type checking
mypy src/
```

### Pre-commit Hooks

Pre-commit hooks will automatically run formatting and linting checks before each commit.

## 📝 Configuration

Key configuration options in `.env`:

- `APP_ENV`: Environment (development/staging/production)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `LLM_PROVIDER`: AI provider (openai/anthropic)
- `TELEGRAM_API_ID`: Telegram API credentials

See `.env.example` for all available options.

## 🐳 Docker Services

The `docker-compose.yml` includes:

- **PostgreSQL**: Primary database
- **Redis**: Caching and session storage
- **Qdrant**: Vector database for semantic search
- **Kafka**: Message queue for event streaming
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **Jaeger**: Distributed tracing

## 📚 Documentation

- [Technical Documentation](DOCUMENTATION.md) - Detailed system architecture and specifications
- [Task Breakdown](TASKS.md) - Implementation roadmap and progress tracking
- [API Documentation](http://localhost:8000/docs) - Interactive API documentation (when running)

## 🤝 Contributing

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## 📄 License

This project is proprietary and confidential.

## 🆘 Support

For issues and questions, please contact the development team.

---

**Status**: 🚧 Under Development - Phase 1: Infrastructure Setup Complete
