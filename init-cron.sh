#!/bin/bash
set -e

echo "Setting up cron container..."

# Wait for MySQL to be available
until MYSQL_PWD=$MYSQL_PASSWORD mysql -h"db" -u"$MYSQL_USER" -e "SELECT 1;" >/dev/null 2>&1; do
  echo "MySQL is unavailable - sleeping"
  sleep 2
done

echo "MySQL is up - proceeding with cron setup"

# Update repositories
apt-get update

# Install cron
apt-get install -y cron

# Create log file and make it writable
touch /var/log/cron.log
chmod 0666 /var/log/cron.log

# Create environment file for cron
env | grep -v "no_proxy" > /etc/environment

# Set up cron job for weekly model retraining (with testing every minute job)
cat > /tmp/crontab << 'EOF'
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PYTHONPATH=/app

# Weekly retraining
0 0 * * 0 cd /app && (echo "[CRON JOB $(date)] Starting scheduled retraining"; python setup.py retrain) >> /var/log/cron.log 2>&1

# Test job (runs every minute)
* * * * * cd /app && (echo "[CRON JOB $(date)] Starting scheduled retraining"; python setup.py retrain) >> /var/log/cron.log 2>&1
EOF

# Apply crontab
crontab /tmp/crontab

# Show the active crontab for debugging
echo "Active crontab:"
crontab -l

# Start cron in foreground with logging
echo "Starting cron in foreground..."
cron -f &

# Follow the log file to see if jobs are executing
tail -f /var/log/cron.log