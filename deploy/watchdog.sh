#!/bin/bash
# Watchdog script: checks if the API is healthy, restarts if not.
# Add to crontab: */5 * * * * /home/arin/trading-bot/deploy/watchdog.sh

HEALTH_URL="http://localhost:8000/api/system/health"
SERVICE="trumpbot-api"
LOG="/home/arin/trading-bot/logs/watchdog.log"

mkdir -p "$(dirname "$LOG")"

response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$HEALTH_URL" 2>/dev/null)

if [ "$response" != "200" ]; then
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Health check failed (HTTP $response), restarting $SERVICE" >> "$LOG"
    sudo systemctl restart "$SERVICE"
else
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - OK" >> "$LOG"
fi

# Trim log to last 500 lines
tail -500 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
