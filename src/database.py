import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()


@dataclass(frozen=True)
class DatabaseConfig:
    user: str
    password: str
    database: str
    host: str
    port: str

    @staticmethod
    def from_env() -> "DatabaseConfig":
        return DatabaseConfig(
            user=os.getenv("POSTGRES_USER", "rag"),
            password=os.getenv("POSTGRES_PASSWORD", "rag"),
            database=os.getenv("POSTGRES_DB", "ragdb"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
        )


class DatabaseConnection:

    def __init__(self, config: DatabaseConfig | None = None):
        self._config = config or DatabaseConfig.from_env()

    def connection_string(self) -> str:
        return (
            "postgresql+psycopg://"
            f"{self._config.user}:{self._config.password}"
            f"@{self._config.host}:{self._config.port}"
            f"/{self._config.database}"
        )