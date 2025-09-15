# Atlas Backend - Healthcare AI Assistant

This is the Python FastAPI backend for the Atlas healthcare AI assistant application. It provides specialized AI capabilities for Kaiser Permanente's healthcare environment, including medical research, data analysis, and conversational support.

## Features

- **Healthcare-Focused AI**: Specialized assistant for medical professionals and patients
- **Streaming Responses**: Real-time chat responses using Server-Sent Events
- **Web Search Integration**: Research capabilities for medical topics using OpenAI tools
- **Data Analysis**: Support for CSV/Excel file analysis with healthcare data insights
- **Robust API**: RESTful endpoints with proper error handling and logging

## Architecture

The backend consists of:
- **Main API** (`main.py`): FastAPI application with chat endpoint
- **Chat Agent** (`agents/chat.py`): Healthcare AI assistant with specialized prompts
- **Streaming Support**: Real-time response streaming for enhanced user experience

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

6. **Run the server:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

The backend will be available at `http://localhost:8000`

## API Endpoints

### POST `/api/chat`
Main chat endpoint for healthcare conversations and data analysis.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "What are the latest treatments for diabetes?"}
  ],
  "selectedChatModel": "gpt-5",
  "requestHints": {}
}
```

**Response:** Server-Sent Events stream with real-time chat responses.

### Health Check
- `GET /` - Basic health check endpoint

## Healthcare Capabilities

- **Medical Research**: Access to web search for current medical information
- **Data Analysis**: Process and analyze healthcare datasets (CSV/Excel)
- **Conversational AI**: Natural language interactions for healthcare queries
- **Kaiser-Specific**: Tailored for Kaiser Permanente's healthcare environment

## Development

### Project Structure
```
backend/
├── main.py              # FastAPI application entry point
├── agents/
│   └── chat.py         # Healthcare AI chat agent
├── requirements.txt    # Python dependencies
├── activate.sh        # Environment activation script
└── README.md          # This file
```

### Key Dependencies
- **FastAPI**: Modern Python web framework
- **OpenAI**: AI model integration for healthcare reasoning
- **python-dotenv**: Environment variable management
- **uvicorn**: ASGI server for FastAPI

## Environment Variables

Create a `.env` file in the backend directory:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Integration with Frontend

The backend integrates seamlessly with the Next.js frontend through:
- RESTful API endpoints
- Server-Sent Events for real-time streaming
- CORS support for cross-origin requests
- Proper error handling and logging

## Contributing

When contributing to the backend:
1. Follow the existing code style and structure
2. Add proper type hints and documentation
3. Test healthcare-specific functionality thoroughly
4. Ensure compliance with healthcare data privacy standards
