# Use the official Python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up user 1000 (Hugging Face Spaces non-root requirement)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements first for better layer caching
COPY --chown=user:user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into container owned by user
COPY --chown=user:user . $HOME/app

# Switch to non-root user
USER user

# Expose port 7860 (Hugging Face default port)
EXPOSE 7860

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Command to run Streamlit on port 7860 with iframe embedding enabled
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
