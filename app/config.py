"""
app/config.py - Central settings loaded from environment / .env file.
Updated to support proxy base URLs and defaults for provided tokens.
"""
from __future__ import annotations
# Load .env file into environment variables BEFORE importing anything else
from dotenv import load_dotenv
load_dotenv()
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM (OpenRouter)
    openrouter_api_key: str = "MISSING"
    openrouter_model: str = "deepseek/deepseek-chat-v3-0324:free"
    # Allow overriding OpenRouter base URL (proxy vs public)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Apify (data)
    apify_token: str = "MISSING"
    apify_base_url: str = "https://api.apify.com/v2"

    # Pipeline
    prediction_history_path: str = "./examples/prediction_history.jsonl"
    log_level: str = "INFO"

    # Kalshi (prediction markets)
    kalshi_base_url: str = "https://api.kalshi.com/trade-api/v2"

    # Polymarket (prediction markets)
    polymarket_base_url: str = "https://gamma-api.polymarket.com"

    # OpenAI-compatible provider (for Hermes)
    openai_api_key: str = "MISSING"
    openai_base_url: str = "https://openrouter.ai/api/v1"


settings = Settings()


# Derive OpenAI-compatible fields from OpenRouter if not set (helps Hermes)
if settings.openai_api_key == "MISSING" and settings.openrouter_api_key != "MISSING":
    settings.openai_api_key = settings.openrouter_api_key
if settings.openai_base_url == "https://openrouter.ai/api/v1" and settings.openrouter_base_url != "https://openrouter.ai/api/v1":
    settings.openai_base_url = settings.openrouter_base_url
