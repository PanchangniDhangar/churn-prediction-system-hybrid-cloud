FROM python:3.12-slim

# Set the working directory to the root of the project
WORKDIR /app

# Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set PYTHONPATH so the 'app' and 'src' folders are discoverable
ENV PYTHONPATH=/app

# Use the PORT environment variable provided by Render
# We use the shell form of CMD to allow variable expansion
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"