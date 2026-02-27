# Pull base image
FROM python:3.12-slim-bullseye

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set work directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    build-essential \
    wget \
    ca-certificates \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6 \
    libfreetype6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall pillow && \
    pip install celery==5.3.6 redis==4.6.0 --force-reinstall && \
    pip check

# Verify installations
RUN pip show marshmallow && \
    python -c "import celery; print(f'Celery {celery.__version__} installed')"

# Download NLTK data (only once Pillow and NLTK are installed)
RUN python -m nltk.downloader \
        punkt \
        punkt_tab \
        averaged_perceptron_tagger \
        averaged_perceptron_tagger_eng \
        stopwords \
    && mkdir -p /usr/share/nltk_data \
    && mv /root/nltk_data/* /usr/share/nltk_data

#Install special character codes for chinese, korean, etc
# Add these lines to your Dockerfile
RUN apt-get update && apt-get install -y \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Update font cache (system-level)
RUN fc-cache -fv

# Rebuild matplotlib's font cache (correct method for newer matplotlib)
RUN python -c "import matplotlib.font_manager as fm; fm._load_fontmanager(try_read_cache=False)"


# Copy project
COPY . .

# Explicitly copy media and staticfiles directories
COPY ./media /code/media
COPY ./staticfiles /code/staticfiles
