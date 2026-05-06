# ---------------- Base Image ----------------
FROM python:3.12-slim

# ---------------- Set Work Directory ----------------
WORKDIR /app

# Prevent Python from writing pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------------- Install System Dependencies ----------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------- Install Python Dependencies ----------------
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------- Copy Project Files ----------------
COPY . .

# ---------------- Expose Port ----------------
EXPOSE 5000

# ---------------- Run App with Gunicorn ----------------
CMD gunicorn -b 0.0.0.0:${PORT:-5000} app:app
