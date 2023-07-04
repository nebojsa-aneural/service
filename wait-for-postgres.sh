#!/bin/sh
# wait-for-postgres.sh

set -e

host="$1"
shift
cmd="$@"

ATTEMPTS=0
MAX_ATTEMPTS=30

until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$host" -U "$POSTGRES_USER" -c '\q' || [ $ATTEMPTS -eq $MAX_ATTEMPTS ]; do
  ATTEMPTS=$((ATTEMPTS + 1))
  >&2 echo "Postgres is unavailable - Attempt $ATTEMPTS out of $MAX_ATTEMPTS, sleeping for 10 seconds"
  sleep 10
done

if [ $ATTEMPTS -eq $MAX_ATTEMPTS ]; then
  >&2 echo "Max attempts reached, Postgres is still unavailable - terminating"
  exit 1
fi

# Grant necessary privileges to the user 'aneural'
PGPASSWORD=$POSTGRES_PASSWORD psql -h "$host" -U "$POSTGRES_USER" -c "GRANT ALL PRIVILEGES ON SCHEMA public TO aneural; GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aneural;"

>&2 echo "Postgres is up - executing command"
exec $cmd