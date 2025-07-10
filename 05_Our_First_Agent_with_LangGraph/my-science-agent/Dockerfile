# ---- Build frontend ----
FROM node:20 AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Backend ----
FROM python:3.11-slim AS backend
WORKDIR /app

# Copia los requirements y el código backend
COPY backend/ ./backend/
COPY my-science-agent/backend/requirements.txt ./backend/requirements.txt

# Instala dependencias del backend
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements.txt

# Copia el build del frontend al backend
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Copia el resto del código necesario (por si hay archivos raíz)
COPY . .

# Variables de entorno para Hugging Face Spaces
ENV PORT=7860
ENV HF_HOME=/data

# Expone el puerto recomendado por Hugging Face Spaces
EXPOSE 7860

# Comando para arrancar FastAPI en el puerto 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"] 