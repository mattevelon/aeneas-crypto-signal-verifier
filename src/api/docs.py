"""
API Documentation configuration and custom descriptions.
"""

from typing import Dict, Any

# API Metadata
API_TITLE = "AENEAS - Crypto Trading Signal Verification System"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
## Overview

AENEAS is an AI-powered cryptocurrency trading signal verification system that analyzes signals from Telegram channels to provide deep analysis, risk assessment, and comprehensive justifications for trading decisions.

## Key Features

- 🚀 **Real-time Signal Detection**: Automated extraction with 50+ regex patterns
- 🤖 **AI-Powered Analysis**: Multi-provider LLM integration (GPT-4, Claude, OpenRouter)
- 📊 **Market Validation**: Real-time price verification and liquidity analysis
- ⚠️ **Risk Assessment**: Kelly Criterion position sizing and VaR calculations
- 🌍 **Multi-language Support**: English, Russian, Chinese, Spanish
- ⚡ **High Performance**: <2-second processing with 99.9% uptime

## Architecture

The system uses a 5-layer microservices-inspired architecture:

1. **Data Ingestion Layer**: Telegram collector, image processor with OCR
2. **Storage Layer**: PostgreSQL, Redis, Qdrant vector database
3. **Processing Core**: Signal detection, context management, AI integration
4. **Validation Layer**: Market validation, risk assessment, manipulation detection
5. **API Layer**: RESTful endpoints, WebSocket support, authentication

## Authentication

The API uses JWT bearer token authentication. To access protected endpoints:

1. Register a new user account via `/api/v1/auth/register`
2. Login to receive access and refresh tokens via `/api/v1/auth/login`
3. Include the access token in the Authorization header: `Bearer YOUR_TOKEN`
4. Refresh expired tokens via `/api/v1/auth/refresh`

## Rate Limiting

- Default: 60 requests/minute per user
- Burst allowance: 100 requests
- Rate limit headers included in all responses

## WebSocket Support

Real-time updates available via WebSocket at `/ws/v2/connect`:

- Signal updates
- Alerts and notifications
- Performance metrics
- Channel statistics

## Response Format

All API responses follow a consistent format:

```json
{
    "status": "success|error",
    "data": {}, // Response data
    "message": "Optional message",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## Error Handling

Errors are returned with appropriate HTTP status codes and detailed messages:

- 400: Bad Request - Invalid input parameters
- 401: Unauthorized - Missing or invalid authentication
- 403: Forbidden - Insufficient permissions
- 404: Not Found - Resource does not exist
- 429: Too Many Requests - Rate limit exceeded
- 500: Internal Server Error - Server-side error

## Support

For API support and questions, contact the development team.
"""

# Custom OpenAPI tags
TAGS_METADATA = [
    {
        "name": "Authentication",
        "description": "User authentication and authorization endpoints",
    },
    {
        "name": "Signals",
        "description": "Trading signal management and retrieval",
    },
    {
        "name": "Statistics",
        "description": "System and performance statistics",
    },
    {
        "name": "Channels",
        "description": "Telegram channel management",
    },
    {
        "name": "Performance",
        "description": "Performance tracking and analytics",
    },
    {
        "name": "Feedback",
        "description": "User feedback collection and analysis",
    },
    {
        "name": "Collector",
        "description": "Telegram data collection management",
    },
    {
        "name": "Health",
        "description": "System health and status checks",
    },
    {
        "name": "WebSocket",
        "description": "Real-time WebSocket connections",
    },
    {
        "name": "WebSocketV2",
        "description": "Enhanced WebSocket with subscription management",
    }
]

# API Response examples
API_RESPONSES = {
    "SignalExample": {
        "description": "Example signal response",
        "content": {
            "application/json": {
                "example": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "source_channel_id": 12345,
                    "pair": "BTC/USDT",
                    "direction": "long",
                    "entry_price": 45000.00,
                    "stop_loss": 43500.00,
                    "take_profits": [46000, 47000, 48000],
                    "risk_level": "medium",
                    "confidence_score": 85.5,
                    "status": "active",
                    "created_at": "2024-01-15T10:30:00Z",
                    "justification": {
                        "technical": "Strong support at $43,000 with bullish divergence",
                        "fundamental": "Positive market sentiment and institutional buying",
                        "risk": "Position size limited to 2% of portfolio"
                    }
                }
            }
        }
    },
    "ErrorExample": {
        "description": "Example error response",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Signal not found",
                    "status_code": 404,
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }
    },
    "TokenExample": {
        "description": "Example authentication token response",
        "content": {
            "application/json": {
                "example": {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer"
                }
            }
        }
    },
    "StatsExample": {
        "description": "Example statistics response",
        "content": {
            "application/json": {
                "example": {
                    "period_days": 7,
                    "total_signals": 150,
                    "active_signals": 25,
                    "average_confidence": 78.5,
                    "success_rate": 68.5,
                    "performance_metrics": {
                        "total_pnl": 12500.50,
                        "win_rate": 0.685,
                        "sharpe_ratio": 1.85,
                        "max_drawdown": -0.12
                    }
                }
            }
        }
    }
}

# Custom API examples for specific endpoints
ENDPOINT_EXAMPLES = {
    "/api/v1/signals": {
        "GET": {
            "summary": "List all signals with optional filters",
            "description": """
            Retrieve a paginated list of trading signals with various filtering options.
            
            Filters can be combined to narrow down results. Results are ordered by creation time (newest first).
            """,
            "parameters": [
                {
                    "name": "pair",
                    "description": "Filter by trading pair (e.g., BTC/USDT)",
                    "example": "BTC/USDT"
                },
                {
                    "name": "status",
                    "description": "Filter by signal status",
                    "example": "active"
                },
                {
                    "name": "min_confidence",
                    "description": "Minimum confidence score (0-100)",
                    "example": 70
                },
                {
                    "name": "limit",
                    "description": "Number of results to return",
                    "example": 50
                },
                {
                    "name": "offset",
                    "description": "Number of results to skip",
                    "example": 0
                }
            ]
        },
        "POST": {
            "summary": "Create a new trading signal",
            "description": """
            Submit a new trading signal for verification and analysis.
            
            The signal will be automatically validated against market data and assessed for risk.
            Real-time notifications will be sent to subscribed WebSocket connections.
            """
        }
    },
    "/api/v1/stats/overview": {
        "GET": {
            "summary": "Get system overview statistics",
            "description": """
            Retrieve comprehensive system statistics including signal counts, performance metrics,
            and success rates for a specified time period.
            
            This endpoint aggregates data from multiple sources to provide a complete picture
            of system performance.
            """
        }
    },
    "/ws/v2/connect": {
        "WEBSOCKET": {
            "summary": "Establish WebSocket connection for real-time updates",
            "description": """
            Connect to the WebSocket server for real-time signal updates and notifications.
            
            Connection supports:
            - Automatic reconnection with state recovery
            - Subscription management for different event types
            - Message filtering and priority levels
            - Heartbeat mechanism for connection monitoring
            
            Authentication is optional via query parameter: `?token=YOUR_TOKEN`
            
            Message format:
            ```json
            {
                "type": "subscribe",
                "data": {
                    "type": "signals",
                    "filters": {"pair": "BTC/USDT"},
                    "priority_filter": ["high", "urgent"]
                }
            }
            ```
            """
        }
    }
}

def get_custom_openapi_schema(app):
    """
    Generate custom OpenAPI schema with enhanced documentation.
    """
    from fastapi.openapi.utils import get_openapi
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
        tags=TAGS_METADATA
    )
    
    # Add custom examples
    if "paths" in openapi_schema:
        for path, methods in ENDPOINT_EXAMPLES.items():
            if path in openapi_schema["paths"]:
                for method, details in methods.items():
                    method_lower = method.lower()
                    if method_lower in openapi_schema["paths"][path]:
                        openapi_schema["paths"][path][method_lower].update(details)
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT authentication token obtained from login endpoint"
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    # Add response examples
    openapi_schema["components"]["examples"] = API_RESPONSES
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.aeneas.crypto",
            "description": "Production server"
        }
    ]
    
    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Full system documentation",
        "url": "https://docs.aeneas.crypto"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema
