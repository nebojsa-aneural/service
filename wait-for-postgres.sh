#!/bin/sh
# wait-for-postgres.sh

set -e

host="$1"
shift
cmd="$@"

ATTEMPTS=0
MAX_ATTEMPTS=10

until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$host" -U "$POSTGRES_USER" -c '\q' || [ $ATTEMPTS -eq $MAX_ATTEMPTS ]; do
  ATTEMPTS=$((ATTEMPTS + 1))
  >&2 echo "Postgres is unavailable - Attempt $ATTEMPTS out of $MAX_ATTEMPTS, sleeping for 5 seconds"
  sleep 5
done

if [ $ATTEMPTS -eq $MAX_ATTEMPTS ]; then
  >&2 echo "Max attempts reached, Postgres is still unavailable - terminating"
  exit 1
fi

>&2 echo "Postgres is up - executing command"
exec $cmd
