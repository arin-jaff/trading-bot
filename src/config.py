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

    # Fine-tuning (runs on Pi nightly — Pythia-160M with LoRA)
    fine_tune_enabled: bool = os.getenv('FINE_TUNE_ENABLED', 'true').lower() == 'true'
    fine_tune_model: str = os.getenv('FINE_TUNE_MODEL', 'EleutherAI/pythia-160m')
    fine_tune_lora_rank: int = int(os.getenv('FINE_TUNE_LORA_RANK', '16'))
    fine_tune_epochs: int = int(os.getenv('FINE_TUNE_EPOCHS', '3'))
    fine_tune_max_length: int = int(os.getenv('FINE_TUNE_MAX_LENGTH', '512'))
    fine_tune_batch_size: int = int(os.getenv('FINE_TUNE_BATCH_SIZE', '1'))
    fine_tune_grad_accum: int = int(os.getenv('FINE_TUNE_GRAD_ACCUM', '8'))
    fine_tune_learning_rate: float = float(os.getenv('FINE_TUNE_LR', '5e-4'))
    fine_tune_mc_sims: int = int(os.getenv('FINE_TUNE_MC_SIMS', '200'))
    fine_tune_hour: int = int(os.getenv('FINE_TUNE_HOUR', '2'))  # 2 AM ET nightly
    fine_tune_gradient_checkpointing: bool = os.getenv('FINE_TUNE_GRAD_CKPT', 'true').lower() == 'true'

    # Social media analysis
    social_media_scrape_minutes: int = int(os.getenv('SOCIAL_MEDIA_SCRAPE_MINUTES', '30'))

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

    def validate_fine_tune(self) -> bool:
        """Check if fine-tuning dependencies are available."""
        if not self.fine_tune_enabled:
            return False
        try:
            import torch
            import transformers
            import peft
            return True
        except ImportError:
            return False

    def get_status(self) -> dict:
        return {
            'kalshi_configured': self.validate_kalshi(),
            'llm_configured': self.validate_llm(),
            'youtube_configured': bool(self.youtube_api_key),
            'email_configured': self.validate_email(),
            'fine_tune_configured': self.validate_fine_tune(),
            'fine_tune_model': self.fine_tune_model,
            'database_url': self.database_url,
        }


config = AppConfig()
