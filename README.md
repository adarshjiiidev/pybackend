# Gemini Financial Agent

A production-ready AI agent for financial analysis using Google Gemini, LangGraph, and Pinecone.

## Features
- **Real-time Data**: Fetches market data using `yfinance`.
- **Semantic Memory**: Stores and retrieves insights using Pinecone.
- **Advanced Reasoning**: Uses Gemini Pro for financial analysis.
- **Structured Output**: Delivers professional, structured financial reports.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```
   - `GEMINI_API_KEY`: Get from Google AI Studio.
   - `PINECONE_API_KEY`: Get from Pinecone Console.
   - `PINECONE_INDEX`: Name of your Pinecone index (dimension 768 for Gemini embeddings).

3. **Run**:
   ```bash
   python main.py
   ```

## Architecture
- **LangGraph**: Orchestrates the workflow.
- **Nodes**:
  - `InputParsing`: Understands intent.
  - `DataFetch`: Gets live data.
  - `Embedding`: Stores data in Pinecone.
  - `Retrieval`: Fetches historical context.
  - `Reasoning`: Analyzes data.
  - `ResponseGeneration`: Formats output.

## Sample Queries
- "What is the current price and PE ratio of AAPL?"
- "Compare the valuation of MSFT and GOOGL."
- "Analyze the risk factors for TSLA based on recent volume."
