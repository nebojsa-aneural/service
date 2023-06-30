-- Connect to the default database
\c postgres;

-- Check if the database 'aneural' exists
SELECT datname FROM pg_database WHERE datname = 'aneural';

-- If the database doesn't exist, create it
SELECT pg_create_database('aneural');

-- Connect to the 'aneural' database
\c aneural;

-- Create user and grant privileges
CREATE USER IF NOT EXISTS aneural WITH ENCRYPTED PASSWORD 'aneural';
GRANT ALL PRIVILEGES ON DATABASE aneural TO aneural;

-- Create the tasks table in the public schema if it doesn't exist
CREATE TABLE IF NOT EXISTS public.tasks (
    "Uuid" UUID PRIMARY KEY,
    "Client" TEXT,
    "ModelPath" TEXT,
    "RequestDateTime" TIMESTAMP,
    "ResponseDateTime" TIMESTAMP,
    "DeadlineDateTime" TIMESTAMP,
    "InputImagePath" TEXT,
    "OutputImagePath" TEXT,
    "ProcessingTime" FLOAT,
    "RunTime" FLOAT,
    "InferenceTime" FLOAT,
    "Status" TEXT,
    "Priority" INT,
    "Resources" JSONB
);