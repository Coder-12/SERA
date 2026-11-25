from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    SERA Configuration Loader (Pydantic v2)
    ---------------------------------------
    Loads environment variables automatically from `.env` file and system envs.
    Field names are automatically matched to UPPERCASE environment variables.
    Example:
        arxiv_api_base  ->  ARXIV_API_BASE
        openai_api_key  ->  OPENAI_API_KEY
        st_model        ->  ST_MODEL
    """

    # === Retrieval ===
    arxiv_api_base: str = Field(
        default="https://export.arxiv.org/api/query",
        description="Base URL for ArXiv API",
    )

    # === Embeddings ===
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (optional)",
    )
    st_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name (used if no API key)",
    )

    # === Vector Store ===
    vector_db: str = Field(
        default="simple",
        description="Vector DB type: 'chroma' or 'simple'",
    )
    chroma_api_url: str | None = Field(
        default=None,
        description="Chroma API endpoint (if using remote server)",
    )
    chroma_collection: str = Field(
        default="sera-research",
        description="Default Chroma collection name",
    )

    # === Logging / General ===
    log_level: str = Field(
        default="INFO",
        description="Global application log level (e.g., DEBUG, INFO, WARNING)",
    )

    # === Model Configuration ===
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Instantiate global settings (import-safe)
settings = Settings()


# if __name__ == "__main__":
#     # Quick debug print (remove in production)
#     print("✅ SERA Configuration Loaded:")
#     print("ARXIV_API_BASE:", settings.arxiv_api_base)
#     print("VECTOR_DB:", settings.vector_db)
