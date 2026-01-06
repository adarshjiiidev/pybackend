# Daddy's AI - Financial Agent API

A multi-layer financial agent for Indian stock market analysis and LTP Calculator concepts.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Development Server
```bash
# Option 1: Using uvicorn directly
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Option 2: Using Python
python api.py
```

### 3. API is live at: `http://localhost:8000`

---

## 📡 API Endpoints

### Health Check
```
GET /
GET /health
```

### Chat (Main Endpoint)
```
POST /chat
Content-Type: application/json

{
  "message": "compare tcs and itc"
}
```

**Response:**
```json
{
  "response": "## Analysis...",
  "intent": "comparative_analysis",
  "language": "english"
}
```

---

## 🔗 Next.js Integration

### Install axios or use fetch
```bash
npm install axios
```

### Create API utility (`lib/api.ts`)
```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ChatResponse {
  response: string;
  intent?: string;
  language?: string;
  error?: string;
}

export async function sendMessage(message: string): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });
  
  if (!res.ok) {
    throw new Error('Failed to send message');
  }
  
  return res.json();
}
```

### Use in React Component
```tsx
'use client';
import { useState } from 'react';
import { sendMessage, ChatResponse } from '@/lib/api';

export default function Chat() {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    setLoading(true);
    try {
      const res = await sendMessage(input);
      setResponse(res);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about stocks..."
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Thinking...' : 'Send'}
        </button>
      </form>
      
      {response && (
        <div>
          <p><strong>Intent:</strong> {response.intent}</p>
          <div dangerouslySetInnerHTML={{ __html: response.response }} />
        </div>
      )}
    </div>
  );
}
```

### Environment Variables (`.env.local`)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🚀 Production Deployment

### Option 1: Railway/Render
1. Push to GitHub
2. Connect to Railway/Render
3. Set environment variables
4. Deploy!

### Option 2: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 3: Manual VPS
```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 🎯 Features

- **Multi-layer reasoning**: AI picks the right response mode
- **Language-aware**: Hindi → WhatsApp style, English → Professional
- **LTP Calculator**: Options trading concepts built-in
- **Fundamental Analysis**: Stock comparisons with profit margins, PE ratios

---

## 📝 Example Queries

| Query | Intent | Language |
|-------|--------|----------|
| "compare tcs and itc" | comparative_analysis | english |
| "wtb kya hota hai" | options_trading | hindi |
| "what is wtb" | options_trading | english |
| "hi" | general_chat | english |
| "nifty ka scenario kya hai" | options_trading | hindi |
