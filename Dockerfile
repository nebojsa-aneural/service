# Base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install the PostgreSQL client library, PostgreSQL client, and build-essential
RUN apt-get update && apt-get install -y libpq-dev postgresql-client gcc python3-dev

# Copy the necessary files to the container
COPY model/ model/
COPY systemd/segmentation.service systemd/system/
COPY models.py .
COPY requirements.txt .
COPY .env .
COPY simple.py .
COPY wait-for-postgres.sh .

# Make the script executable
RUN chmod +x wait-for-postgres.sh

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the application
CMD ["./wait-for-postgres.sh", "db", "sh", "-c", "psql -h db -U postgres -c 'GRANT ALL PRIVILEGES ON SCHEMA public TO aneural;' && python simple.py"]


