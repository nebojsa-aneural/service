-- Connect to the default database
\c postgres;

-- Check if the database 'aneural' exists
SELECT datname FROM pg_database WHERE datname = 'aneural';

-- If the database doesn't exist, create it
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'aneural') THEN
        CREATE DATABASE aneural;
    END IF;
END $$;

-- Connect to the 'aneural' database
\c aneural;

-- Create user if it doesn't exist
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_user WHERE usename = 'aneural') THEN
        CREATE USER aneural WITH ENCRYPTED PASSWORD 'aneural';
    END IF;
END $$;

-- Grant privileges to the user
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