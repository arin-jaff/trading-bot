"""Run the API server with background scheduler."""

import uvicorn
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from src.database.db import init_db
from src.scheduler import create_scheduler
from src.api.server import app


def main():
    # Initialize database
    init_db()
    logger.info("Database initialized")

    # Start scheduler
    scheduler = create_scheduler()
    scheduler.start()
    logger.info("Background scheduler started")

    # Run API server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
