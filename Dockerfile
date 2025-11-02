FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential libgl1 libglib2.0-0 poppler-utils libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
# copy app code
COPY . /app
# explicitly copy service account key
COPY docai-accessor-key.json /app/docai-accessor-key.json


ENV PORT=8000
EXPOSE 8000

CMD ["python", "allfinal.py"]
