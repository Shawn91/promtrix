from dataclasses import dataclass

from dotenv import dotenv_values

env = dotenv_values()


@dataclass
class Config:
    OPENROUTER_API_KEY: str = env.get("OPENROUTER_API_KEY")
