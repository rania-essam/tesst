# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable for Hugging Face cache
ENV TRANSFORMERS_CACHE=/app/cache

# Ensure cache and model directories exist and have the right permissions
RUN mkdir -p /app/cache /app/model /app && chmod -R 777 /app/cache /app/model /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies, including gunicorn
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the application code
COPY . .



# Set environment variable for Flask
ENV PORT 7860

# Expose port 7860
EXPOSE 7860

# Run the application with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
