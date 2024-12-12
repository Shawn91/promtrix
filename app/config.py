from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

env = dotenv_values()

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class Config:
    OPENROUTER_API_KEY: str = env.get("OPENROUTER_API_KEY")
    ENV: str = env.get("ENVIRONMENT", "dev")
    # todo: allow users to specify their own database path
    DATABASE_PATH: str = (
        PROJECT_ROOT / "experiments" / "database.sqlite"
        if env.get("ENVIRONMENT", "development") == "development"
        else ""
    )
