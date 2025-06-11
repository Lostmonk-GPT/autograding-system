import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

class Config:
    """Configuration manager for the AutoGrading system."""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
        """
        # Load environment variables
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
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def _validate_config(self):
        """Validate required configuration values."""
        # Only validate API keys if not in offline/mock mode
        if not self.get_env('OFFLINE_MODE', 'false').lower() == 'true':
            required_env_vars = [
                'OPENAI_API_KEY',
            ]
            
            missing_vars = []
            for var in required_env_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
        else:
            print("Running in offline mode - API keys not required")
    
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
    
    def get_env(self, key: str, default: str = None) -> str:
        """Get environment variable value.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value
        """
        return os.getenv(key, default)
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key."""
        return self.get_env('OPENAI_API_KEY')
    
    @property
    def canvas_api_token(self) -> str:
        """Get Canvas API token."""
        return self.get_env('CANVAS_API_TOKEN')
    
    @property
    def canvas_base_url(self) -> str:
        """Get Canvas base URL."""
        return self.get_env('CANVAS_BASE_URL')
    
    @property
    def google_client_id(self) -> str:
        """Get Google client ID."""
        return self.get_env('GOOGLE_CLIENT_ID')
    
    @property
    def google_client_secret(self) -> str:
        """Get Google client secret."""
        return self.get_env('GOOGLE_CLIENT_SECRET')
    
    @property
    def debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get('application.debug', False)

# Global configuration instance
config = Config()
