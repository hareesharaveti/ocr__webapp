# 1. Use an official lightweight Python setup
FROM python:3.9-slim

# 2. Install the required System Tools (Tesseract & Poppler)
# This is the magic step that makes OCR work on the cloud!
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set up the working directory
WORKDIR /app

# 4. Copy requirements and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
COPY . .

# 6. Command to start the server
# Render automatically provides the PORT variable
CMD gunicorn app:app --bind 0.0.0.0:$PORT