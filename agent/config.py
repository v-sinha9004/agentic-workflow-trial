import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"


def _load_dotenv_fallback(env_path: Path) -> None:
    """Simple built-in .env parser that doesn't require third-party dependencies."""
    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                # Do not overwrite existing environment variables
                if key not in os.environ:
                    os.environ[key] = val


def load_config(model_override: str | None = None) -> Config:
    """
    Loads configuration from environment variables or .env file.
    Priority:
      1. Code argument (model_override)
      2. Environment variables
      3. .env file
    """
    # Look for .env in current directory or project root
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        env_file = Path(__file__).resolve().parent.parent / ".env"

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file)
    except ImportError:
        _load_dotenv_fallback(env_file)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = model_override or os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")

    return Config(
        api_key=api_key,
        model=model or "gpt-4o-mini",
        base_url=base_url or "https://api.openai.com/v1",
    )
