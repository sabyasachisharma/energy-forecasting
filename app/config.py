"""
Configuration management for the energy pricing application.
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    """TimescaleDB Tiger Cloud configuration."""
    host: str
    port: int
    name: str
    user: str
    password: str
    ssl_mode: str = "require"
    
    @property
    def connection_url(self) -> str:
        """Get the database connection URL with SSL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}?sslmode={self.ssl_mode}"
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create Tiger Cloud database config from environment variables."""
        return cls(
            host=os.getenv("DB_HOST", "jp4ibi3bkd.wg59k3p2y1.tsdb.cloud.timescale.com"),
            port=int(os.getenv("DB_PORT", "37813")),
            name=os.getenv("DB_NAME", "tsdb"),
            user=os.getenv("DB_USER", "tsdbadmin"),
            password=os.getenv("DB_PASSWORD"),
            ssl_mode=os.getenv("DB_SSL_MODE", "require")
        )

@dataclass
class AppConfig:
    """Application configuration settings."""
    database: DatabaseConfig
    use_database: bool
    index_dir: str
    api_host: str
    api_port: int
    openai_api_key: Optional[str]
    openai_model: str
    embeddings_model: str
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create app config from environment variables."""
        return cls(
            database=DatabaseConfig.from_env(),
            use_database=os.getenv("USE_DATABASE", "true").lower() == "true",
            index_dir=os.getenv("INDEX_DIR", "data"),
            api_host=os.getenv("API_HOST", "127.0.0.1"),
            api_port=int(os.getenv("API_PORT", "8000")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            embeddings_model=os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        issues = []
        
        if self.use_database:
            if not self.database.password or len(self.database.password) < 8:
                issues.append("Database password not set or too weak")
            if not self.database.user:
                issues.append("Database user not set")
            if not self.database.name:
                issues.append("Database name not set")
            if not self.database.host:
                issues.append("Database host not set")
        
        
        if issues:
            logging.warning("Configuration issues found:")
            for issue in issues:
                logging.warning(f"  - {issue}")
            return False
            
        return True

_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get the global application configuration."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
        logging.info("Configuration loaded from environment variables")
        
        if not _config.validate():
            logging.warning("Configuration validation failed - some features may not work properly")
    
    return _config

def reload_config() -> AppConfig:
    """Reload configuration from environment variables."""
    global _config
    load_dotenv(override=True)
    _config = None
    return get_config()
def get_database_url() -> str:
    """Get database connection URL."""
    return get_config().database.connection_url

def use_database() -> bool:
    """Check if database should be used."""
    return get_config().use_database


def get_index_dir() -> str:
    """Get index directory path."""
    return get_config().index_dir

def get_openai_config() -> tuple[Optional[str], str]:
    """Get OpenAI API key and model."""
    config = get_config()
    return config.openai_api_key, config.openai_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
