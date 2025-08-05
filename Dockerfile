FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS server

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Combine apt operations for better caching and faster builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-setuptools python3-dev \
    build-essential curl ca-certificates \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY server/requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --prefer-binary --no-compile --no-deps \
    -r requirements.txt \
    && pip install --no-compile \
    -r requirements.txt \
    && rm -rf /root/.cache/pip/* /tmp/*

# Copy application code last for better caching
COPY server/ .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "__init__:create_app()"]
