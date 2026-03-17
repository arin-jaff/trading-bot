#!/bin/bash
# =============================================================================
# TrumpGPT Raspberry Pi Setup Script
# Run this ON the Raspberry Pi after cloning the repo.
# Usage: cd /home/pi/trading-bot && bash deploy/setup-pi.sh
# =============================================================================

set -e

echo "=========================================="
echo "  TrumpGPT Raspberry Pi Setup"
echo "=========================================="

# --- System dependencies ---
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv python3-dev \
    libffi-dev libssl-dev libopenblas-dev git curl

# --- Python virtual environment ---
echo "[2/7] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# --- Install Python dependencies (lightweight set) ---
echo "[3/7] Installing Python dependencies (Pi-optimized)..."
pip install --upgrade pip setuptools wheel
pip install -r requirements-pi.txt

# --- Download NLTK data ---
echo "[4/7] Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

# --- Initialize database ---
echo "[5/7] Initializing database..."
python3 -c "from src.database.db import init_db; init_db()"

# --- Install systemd services ---
echo "[6/7] Installing systemd services..."
sudo cp deploy/trumpbot-api.service /etc/systemd/system/
sudo cp deploy/trumpbot-gui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trumpbot-api
sudo systemctl enable trumpbot-gui

# --- Install watchdog cron ---
echo "[7/7] Setting up watchdog cron..."
chmod +x deploy/watchdog.sh
mkdir -p logs

# Add watchdog to crontab if not already there
(crontab -l 2>/dev/null | grep -v 'watchdog.sh'; echo "*/5 * * * * /home/pi/trading-bot/deploy/watchdog.sh") | crontab -

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Copy your .env file:  scp .env pi@<pi-ip>:/home/pi/trading-bot/.env"
echo "  2. Copy your secrets/:   scp -r secrets/ pi@<pi-ip>:/home/pi/trading-bot/secrets/"
echo "  3. Start the bot:        sudo systemctl start trumpbot-api"
echo "  4. Start the dashboard:  sudo systemctl start trumpbot-gui"
echo "  5. Check status:         sudo systemctl status trumpbot-api"
echo "  6. View logs:            journalctl -u trumpbot-api -f"
echo ""
echo "Dashboard will be available at: http://<pi-ip>:8501"
echo "API will be available at:       http://<pi-ip>:8000"
echo ""
