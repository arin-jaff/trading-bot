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

    # Gemini (for news enrichment)
    gemini_api_key: str = os.getenv('GEMINI_API_KEY', '')

    # Email notifications
    email_enabled: bool = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
    email_smtp_host: str = os.getenv('EMAIL_SMTP_HOST', 'smtp.gmail.com')
    email_smtp_port: int = int(os.getenv('EMAIL_SMTP_PORT', '587'))
    email_from: str = os.getenv('EMAIL_FROM', '')
    email_app_password: str = os.getenv('EMAIL_APP_PASSWORD', '')
    email_to: str = os.getenv('EMAIL_TO', '')

    # Pipeline mode ('local' for Pi, 'colab' for Colab+Drive)
    pipeline_mode: str = os.getenv('PIPELINE_MODE', 'local')
    markov_order: int = int(os.getenv('MARKOV_ORDER', '3'))
    monte_carlo_simulations: int = int(os.getenv('MONTE_CARLO_SIMULATIONS', '2000'))
    retrain_interval_hours: int = int(os.getenv('RETRAIN_INTERVAL_HOURS', '6'))

    # App
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    refresh_interval: int = int(os.getenv('REFRESH_INTERVAL_SECONDS', '300'))
    api_port: int = int(os.getenv('API_PORT', '8000'))
    gui_port: int = int(os.getenv('GUI_PORT', '8501'))

    def validate_kalshi(self) -> bool:
        return bool(self.kalshi_email and self.kalshi_password)

    def validate_llm(self) -> bool:
        return bool(self.openai_api_key or self.anthropic_api_key)

    def validate_email(self) -> bool:
        return bool(self.email_enabled and self.email_from and self.email_app_password and self.email_to)

    def get_status(self) -> dict:
        return {
            'kalshi_configured': self.validate_kalshi(),
            'llm_configured': self.validate_llm(),
            'youtube_configured': bool(self.youtube_api_key),
            'email_configured': self.validate_email(),
            'database_url': self.database_url,
        }


config = AppConfig()
