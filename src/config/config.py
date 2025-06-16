"""
Configuration management for AutoGrading system.
Handles YAML configuration files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Configuration manager for the AutoGrading system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Set default config file path
        if config_file is None:
            config_file = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        
        self.config_file = Path(config_file)
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                return config or {}
            else:
                print(f"⚠️  Configuration file not found: {self.config_file}")
                print("Using default configuration values.")
                return self._get_default_config()
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration when file is missing."""
        return {
            "application": {
                "name": "AutoGrading System",
                "version": "1.0.0",
                "debug": True,
                "offline_mode": True
            },
            "ocr": {
                "engines": {
                    "tesseract": {"enabled": True},
                    "easyocr": {"enabled": True}
                },
                "confidence_threshold": 0.7
            },
            "grading": {
                "mock_mode": True,
                "subjects": ["algebra1", "ap_calculus", "geometry"]
            },
            "integrations": {
                "canvas": {"mock_mode": True},
                "google_drive": {"mock_mode": True}
            }
        }
    
    def _validate_config(self):
        """Validate required configuration values."""
        # Only validate API keys if not in offline/mock mode
        offline_mode = self.get_env('OFFLINE_MODE', 'true').lower() == 'true'
        
        if not offline_mode and not self.get('grading.mock_mode', True):
            required_env_vars = ['OPENAI_API_KEY']
            
            missing_vars = []
            for var in required_env_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
        else:
            print("🔧 Running in offline/mock mode - API keys not required")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key.
        
        Args:
            key: Configuration key in dot notation (e.g., 'ocr.confidence_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value
        """
        return os.getenv(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot notation key.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    @property
    def debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get('application.debug', True)
    
    @property
    def offline_mode(self) -> bool:
        """Check if offline mode is enabled."""
        return self.get('application.offline_mode', True)
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self.get_env('OPENAI_API_KEY')
    
    @property
    def canvas_api_token(self) -> Optional[str]:
        """Get Canvas API token."""
        return self.get_env('CANVAS_API_TOKEN')
    
    @property
    def canvas_base_url(self) -> Optional[str]:
        """Get Canvas base URL."""
        return self.get_env('CANVAS_BASE_URL')
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """Get OCR configuration settings."""
        return self.get('ocr', {})
    
    def get_grading_config(self) -> Dict[str, Any]:
        """Get grading configuration settings."""
        return self.get('grading', {})
    
    def is_mock_mode(self, service: str) -> bool:
        """Check if a service is running in mock mode.
        
        Args:
            service: Service name (e.g., 'canvas', 'google_drive', 'grading')
            
        Returns:
            True if service is in mock mode
        """
        if service == 'grading':
            return self.get('grading.mock_mode', True)
        else:
            return self.get(f'integrations.{service}.mock_mode', True)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config(config_file: Optional[str] = None) -> Config:
    """Reload configuration from file.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Reloaded configuration instance
    """
    global config
    config = Config(config_file)
    return config