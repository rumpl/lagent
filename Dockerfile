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

LABEL com.docker.agent.packaging.version="v0.0.2"
LABEL com.docker.agent.runtime="langchain"
LABEL com.docker.agent.secrets.models="OPENAI_API_KEY"

LABEL org.opencontainers.image.author="rumpl"
LABEL org.opencontainers.image.created="2025-09-16T09:17:39Z"
LABEL org.opencontainers.image.description="Demo of Docker Agent Engine with Langchain"

# Run the application
ENV PYTHONPATH=/app
ENTRYPOINT ["python"]
CMD ["-m", "sample_agent.demo"]
