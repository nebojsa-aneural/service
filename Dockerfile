# Base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install the PostgreSQL client library
RUN apt-get update && apt-get install -y libpq-dev

# Copy the necessary files to the container
COPY model/ model/
COPY systemd/segmentation.service systemd/
COPY models.py .
COPY requirements.txt .
COPY simple.py .
COPY .env .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to load environment variables from .env file and run the application
CMD ["/bin/bash", "-c", "source .env && python simple.py"]
