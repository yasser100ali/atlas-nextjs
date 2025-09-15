<a href="https://chat.vercel.ai/">
  <img alt="Atlas - Healthcare AI Assistant" src="app/(chat)/opengraph-image.png">
  <h1 align="center">Atlas - Healthcare AI Assistant</h1>
</a>

<p align="center">
    Atlas is a specialized AI assistant for Kaiser Permanente, built with Next.js and FastAPI, designed to provide healthcare insights, data analysis, and support for medical professionals and patients.
</p>

<p align="center">
  <a href="https://chat-sdk.dev"><strong>Read Docs</strong></a> ·
  <a href="#features"><strong>Features</strong></a> ·
  <a href="#model-providers"><strong>Model Providers</strong></a> ·
  <a href="#deploy-your-own"><strong>Deploy Your Own</strong></a> ·
  <a href="#running-locally"><strong>Running locally</strong></a>
</p>
<br/>

## Features

- **Healthcare-Focused AI Assistant**
  - Specialized for Kaiser Permanente's healthcare environment
  - Research and analysis capabilities for medical topics
  - Support for both general healthcare conversations and specialized medical inquiries
- **Advanced Data Analysis**
  - CSV and Excel file upload and analysis capabilities
  - Healthcare data processing and insights generation
  - Interactive data visualization and reporting
- **Dual Architecture**
  - [Next.js](https://nextjs.org) frontend with App Router for modern web experience
  - Python FastAPI backend for robust AI processing and data analysis
  - Real-time streaming responses for enhanced user experience
- **AI Integration**
  - [AI SDK](https://ai-sdk.dev/docs/introduction) with unified API for LLMs
  - OpenAI GPT-5 integration for advanced healthcare reasoning
  - Web search capabilities for comprehensive medical research
- **Modern UI/UX**
  - [shadcn/ui](https://ui.shadcn.com) components with [Tailwind CSS](https://tailwindcss.com)
  - Dark/light theme support with system preference detection
  - Responsive design optimized for healthcare professionals
- **Data Management**
  - [Neon Serverless Postgres](https://vercel.com/marketplace/neon) for secure chat history
  - [Vercel Blob](https://vercel.com/storage/blob) for medical file storage
  - Secure authentication with [Auth.js](https://authjs.dev)

## Model Providers

This template uses the [Vercel AI Gateway](https://vercel.com/docs/ai-gateway) to access multiple AI models through a unified interface. The default configuration includes [xAI](https://x.ai) models (`grok-2-vision-1212`, `grok-3-mini`) routed through the gateway.

### AI Gateway Authentication

**For Vercel deployments**: Authentication is handled automatically via OIDC tokens.

**For non-Vercel deployments**: You need to provide an AI Gateway API key by setting the `AI_GATEWAY_API_KEY` environment variable in your `.env.local` file.

With the [AI SDK](https://ai-sdk.dev/docs/introduction), you can also switch to direct LLM providers like [OpenAI](https://openai.com), [Anthropic](https://anthropic.com), [Cohere](https://cohere.com/), and [many more](https://ai-sdk.dev/providers/ai-sdk-providers) with just a few lines of code.

## Deploy Your Own

## Architecture Overview

Atlas uses a modern full-stack architecture:

### Frontend (Next.js)
- Built with Next.js 15 and React 19
- App Router for optimal performance and SEO
- Real-time chat interface with streaming responses
- File upload support for CSV/Excel analysis
- Responsive design with dark/light theme support

### Backend (Python FastAPI)
- Healthcare-focused AI assistant using OpenAI GPT-5
- Streaming API responses for real-time chat
- Web search integration for medical research
- Data analysis capabilities for healthcare datasets
- RESTful API design with proper error handling

### Key Components
- **Authentication**: Secure user management with NextAuth.js
- **Database**: PostgreSQL with Drizzle ORM for chat history
- **File Storage**: Vercel Blob for secure medical file storage
- **AI Integration**: OpenAI API with healthcare-specific prompts
- **UI Components**: shadcn/ui with Tailwind CSS for modern interface

## Deploy Your Own

You can deploy your own version of Atlas to Vercel with one click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel%2Fai-chatbot&env=AUTH_SECRET&envDescription=Generate%20a%20random%20secret%20to%20use%20for%20authentication&envLink=https%3A%2F%2Fgenerate-secret.vercel.app%2F32&project-name=atlas-healthcare&repository-name=atlas-healthcare&demo-title=Atlas%20-%20Healthcare%20AI%20Assistant&demo-description=A%20specialized%20AI%20assistant%20for%20Kaiser%20Permanente%20built%20with%20Next.js%20and%20FastAPI&demo-url=https%3A%2F%2Fchat.vercel.ai&products=%5B%7B%22type%22%3A%22integration%22%2C%22protocol%22%3A%22storage%22%2C%22productSlug%22%3A%22neon%22%2C%22integrationSlug%22%3A%22neon%22%7D%2C%7B%22type%22%3A%22blob%22%7D%5D)

## Running locally

Atlas consists of both a Next.js frontend and a Python FastAPI backend. You'll need to set up environment variables for both components.

### Prerequisites

- Node.js 18+ and pnpm
- Python 3.8+
- OpenAI API key

### Environment Setup

1. **Clone the repository and navigate to the project directory**
2. **Install Vercel CLI:** `npm i -g vercel`
3. **Link with Vercel:** `vercel link` (creates `.vercel` directory)
4. **Download environment variables:** `vercel env pull`

### Frontend Setup (Next.js)

```bash
# Install dependencies
pnpm install

# Start the frontend (includes backend via concurrently)
pnpm dev
```

### Backend Setup (Python FastAPI)

If you need to run the backend separately:

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_openai_api_key_here

# Run the backend server
uvicorn main:app --reload --port 8000
```

### Full Development Environment

For the complete development experience with both frontend and backend:

```bash
# This runs both frontend (port 3000) and backend (port 8000)
pnpm dev
```

Your Atlas application will be available at [localhost:3000](http://localhost:3000).

> **Security Note:** Never commit your `.env` file or expose your OpenAI API key in version control.
