# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure Poetry: don't create virtual environment, install dependencies globally
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set work directory
WORKDIR /app

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-root \
    && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY sample_agent/ ./sample_agent/
COPY langgraph.json ./

# Expose port
EXPOSE 8000

# Set default environment variables
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/cagent/agents || exit 1

# Run the application
CMD ["python", "-m", "sample_agent.demo"]
