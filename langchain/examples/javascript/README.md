# LangChain TypeScript Examples

This directory contains TypeScript code examples from the OSS AI Summit: Building with LangChain event.

## Prerequisites

- [Node.js](https://nodejs.org/) 22 or later
- An Azure OpenAI API key

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/microsoft/oss-ai-summit.git
cd oss-ai-summit/langchain/examples/javascript
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Configure Environment Variables

Create a `.env` file in this directory:

```bash
# For Azure OpenAI
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

### 4. Run Examples

```bash
# Run a specific example
tsx 02-create-agent.ts
```
