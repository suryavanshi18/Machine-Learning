FROM python:3.12-slim

WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy remaining files
COPY . .

EXPOSE 7562

CMD ["python", "main.py"]