# Daaddys AI - Autonomous Financial AI Agent

> **Production-grade MVP** multi-agent system for Indian stock markets and global crypto markets using LangGraph, Groq, and FastAPI.

## 🏗️ Architecture

**Multi-Agent System** powered by LangGraph with 6 specialized agents:
- **Router Agent**: Intent classification and mode selection (uses **llama-3.1-8b-instant** for speed)
- **Market Research Agent**: Deep fundamental analysis for Indian equities (uses **llama-3.3-70b-versatile**)
- **Real-time Analysis Agent**: Intraday movements and technical signals (uses **llama-3.3-70b-versatile**)
- **Portfolio Intelligence Agent**: Asset allocation and risk management (uses **llama-3.3-70b-versatile**)
- **Explainer Agent**: Educational content for retail investors (uses **llama-3.1-70b-versatile** with high creativity)
- **Crypto Intelligence Agent**: Cryptocurrency analysis (BTC, ETH, altcoins) (uses **llama-3.3-70b-versatile**)

**Multi-Model System:**
- 🚀 **Fast Routing**: 8B instant model for 60% faster classification
- 🧠 **Deep Reasoning**: 70B versatile models for complex analysis
- 🎨 **Creative Explanations**: Higher temperature for educational content
- ⚡ **Optimized Performance**: Best model for each specific task

**Tech Stack:**
- LangGraph for agent orchestration
- Groq (llama-3.3-70b-versatile) for LLM inference
- FastAPI with Server-Sent Events for streaming
- MongoDB for conversation persistence and caching
- Yahoo Finance API for market data

## 📋 Prerequisites

- Python 3.11+
- MongoDB (local or Atlas)
- Groq API key ([get one here](https://console.groq.com))

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to backend directory
cd c:\Users\Ai\daddys_ai\backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example environment file
copy .env.example .env

# Edit .env and add your Groq API key
# GROQ_API_KEY=your_actual_api_key_here
```

### 3. Start MongoDB

```bash
# Local MongoDB
mongod --dbpath C:\data\db

# OR use MongoDB Atlas (cloud)
# Just configure MONGODB_URL in .env with your connection string
```

### 4. Run the Server

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Server will start at: `http://localhost:8000`

## 📡 API Endpoints

### Chat (Streaming)

```bash
POST /chat/stream

Body:
{
  "query": "Analyze Reliance Industries stock",
  "mode": "auto",  # auto, market_research, realtime_analysis, portfolio, explainer, crypto
  "session_id": "optional-session-id"
}

Response: Server-Sent Events stream
```

### Session Management

```bash
# Create session
POST /chat/session

# Get conversation history
GET /chat/history/{session_id}

# Delete session
DELETE /chat/session/{session_id}
```

### Health Check

```bash
GET /health
```

## 🧪 Testing

### Using cURL

```bash
# Streaming chat
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What is PE ratio?\", \"mode\": \"explainer\"}"

# Create session
curl -X POST http://localhost:8000/chat/session

# Health check
curl http://localhost:8000/health
```

### Using Python

```python
import requests
import json

# Streaming request
response = requests.post(
    "http://localhost:8000/chat/stream",
    json={"query": "Analyze TCS stock", "mode": "market_research"},
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode().replace("data: ", ""))
        print(data["content"], end="", flush=True)
```

## 🎯 Agent Modes

| Mode | Description | Use Cases |
|------|-------------|-----------|
| `auto` | Router automatically selects best agent | General queries |
| `market_research` | Deep fundamental analysis | Long-term investments, company research |
| `realtime_analysis` | Intraday technical analysis | Day trading, short-term signals |
| `portfolio` | Asset allocation strategies | Portfolio construction, diversification |
| `explainer` | Educational content | Learning finance concepts |
| `crypto` | Cryptocurrency analysis | Bitcoin, Ethereum, altcoins |

## 🗂️ Project Structure

```
backend/
├── app/
│   ├── agents/          # 6 specialized agents
│   ├── config/          # Settings and database connection
│   ├── database/        # Repositories and caching
│   ├── graph/           # LangGraph workflow
│   ├── models/          # Pydantic models (API, DB, State)
│   ├── tools/           # Yahoo Finance and formatting utilities
│   ├── utils/           # SSE streaming utilities
│   └── main.py          # FastAPI application
├── requirements.txt
├── .env.example
└── README.md
```

## 🔧 Configuration

Key environment variables:

- `GROQ_API_KEY`: Your Groq API key (required)
- `MONGODB_URL`: MongoDB connection string

**Multi-Model Configuration:**
- `MODEL_REASONING`: Deep analysis model (default: llama-3.3-70b-versatile)
- `MODEL_FAST`: Quick routing model (default: llama-3.1-8b-instant)
- `MODEL_ANALYSIS`: Technical analysis model (default: llama-3.3-70b-versatile)
- `MODEL_CREATIVE`: Educational content model (default: llama-3.1-70b-versatile)
- `MODEL_ROUTER`: Intent classification model (default: llama-3.1-8b-instant)

**Model Parameters:**
- `TEMPERATURE_REASONING`: 0.7 (balanced for analysis)
- `TEMPERATURE_FAST`: 0.3 (low for classification)
- `TEMPERATURE_CREATIVE`: 0.8 (higher for explanations)
- `MAX_TOKENS_REASONING`: 4096
- `MAX_TOKENS_FAST`: 1024

See [MULTI_MODEL_SYSTEM.md](./MULTI_MODEL_SYSTEM.md) for detailed model selection guide.

## 📊 MongoDB Collections

- `conversation_messages`: Chat history
- `user_sessions`: Session management
- `market_data_cache`: Yahoo Finance data caching

## 🎨 Features

✅ **Multi-agent orchestration** with LangGraph  
✅ **Multi-model system** - Optimal model for each task (8B for routing, 70B for analysis)  
✅ **Streaming responses** via SSE  
✅ **Conversation persistence** with MongoDB  
✅ **Market data caching** (5-min TTL)  
✅ **Intent classification** and entity extraction  
✅ **India-specific** context (NSE/BSE, tax awareness)  
✅ **Error handling** and graceful degradation  
✅ **Production-ready** logging and monitoring  

## ⚠️ Important Notes

- **Yahoo Finance**: Free tier with rate limits. Use `.NS` suffix for NSE stocks (e.g., `RELIANCE.NS`)
- **Crypto Regulations**: India has regulatory uncertainty around crypto. Agent includes appropriate disclaimers.
- **Not Financial Advice**: All responses include disclaimers. This is educational content only.

## 🐛 Troubleshooting

**MongoDB connection error:**
```bash
# Ensure MongoDB is running
mongod --version

# Check connection string in .env
```

**Groq API error:**
```bash
# Verify API key is set correctly
# Check quota limits at console.groq.com
```

**Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📝 Development

```bash
# Run with auto-reload
uvicorn app.main:app --reload

# Run with specific host/port
uvicorn app.main:app --host 0.0.0.0 --port 8000

# View logs
# Check terminal output or configure file logging in settings.py
```

## 🚀 Production Deployment

For production:
1. Set `ENVIRONMENT=production` in `.env`
2. Use MongoDB Atlas (cloud)
3. Configure proper CORS origins
4. Use production-grade ASGI server (Gunicorn + Uvicorn)
5. Set up monitoring and alerts
6. Enable rate limiting

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Support

For issues or questions, please refer to the documentation or contact support.

---

**Built with ❤️ for Indian retail investors**
