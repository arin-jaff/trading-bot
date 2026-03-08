"""Centralized configuration management."""

import os
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class AppConfig(BaseModel):
    """Application configuration loaded from environment."""

    # Kalshi
    kalshi_api_key: str = os.getenv('KALSHI_API_KEY', '')
    kalshi_email: str = os.getenv('KALSHI_EMAIL', '')
    kalshi_password: str = os.getenv('KALSHI_PASSWORD', '')
    kalshi_base_url: str = os.getenv('KALSHI_BASE_URL', 'https://trading-api.kalshi.com/trade-api/v2')

    # LLM
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    anthropic_api_key: str = os.getenv('ANTHROPIC_API_KEY', '')

    # Database
    database_url: str = os.getenv('DATABASE_URL', 'sqlite:///data/trading_bot.db')

    # YouTube
    youtube_api_key: str = os.getenv('YOUTUBE_API_KEY', '')

    # App
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    refresh_interval: int = int(os.getenv('REFRESH_INTERVAL_SECONDS', '300'))
    api_port: int = int(os.getenv('API_PORT', '8000'))
    gui_port: int = int(os.getenv('GUI_PORT', '8501'))

    def validate_kalshi(self) -> bool:
        return bool(self.kalshi_email and self.kalshi_password)

    def validate_llm(self) -> bool:
        return bool(self.openai_api_key or self.anthropic_api_key)

    def get_status(self) -> dict:
        return {
            'kalshi_configured': self.validate_kalshi(),
            'llm_configured': self.validate_llm(),
            'youtube_configured': bool(self.youtube_api_key),
            'database_url': self.database_url,
        }


config = AppConfig()
