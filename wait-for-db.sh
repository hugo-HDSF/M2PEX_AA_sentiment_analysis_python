#!/bin/bash
set -e

host="$1"
shift
cmd="$@"

echo "Waiting for MySQL database at $host..."

# Loop until we can connect to MySQL
until MYSQL_PWD=$MYSQL_PASSWORD mysql -h"$host" -u"$MYSQL_USER" -e "SELECT 1;" >/dev/null 2>&1; do
  echo "MySQL is unavailable - sleeping"
  sleep 2
done

echo "MySQL is up - executing command: $cmd"
exec $cmd