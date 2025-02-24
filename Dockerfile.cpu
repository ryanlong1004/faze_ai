# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

# Create app directory and non-root user
WORKDIR /app
RUN useradd -m -u 1000 appuser

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv==0.5.21

# Copy source code and set permissions
COPY . /app
RUN chown -R appuser:appuser /app

USER appuser

# Install dependencies
RUN uv sync

# Set the entrypoint
ENTRYPOINT ["sh", "-c", "python -m src.main $(echo \"$@\" | sed 's/--model[s]* [^ ]* //g')", "--"]
