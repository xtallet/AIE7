# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files explicitly
COPY app.py .
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY README.md .

# Debug: List files to verify copying
RUN ls -la /app
RUN ls -la /app/backend

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Build the frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Copy built frontend to backend static directory
RUN mkdir -p /app/backend/static
RUN cp -r dist/* /app/backend/static/

# Copiar solo la imagen advanced_build_graph.png al directorio de estáticos
RUN cp /app/frontend/public/advanced_build_graph.png /app/backend/static/advanced_build_graph.png

# Set working directory back to app
WORKDIR /app

# Expose port (Hugging Face Spaces will set the actual port)
EXPOSE 7860

# Start the application using the main app.py
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"] 