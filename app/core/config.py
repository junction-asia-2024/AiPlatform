from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "mysql+aiomysql://root:@127.0.0.1:3307/fastapi_sql"

    class Config:
        env_file: str = ".env"
        case_sensitive = True  # Set this according to your case sensitivity needs


settings = Settings()
