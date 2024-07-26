from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DB_DATABASE: str
    DB_PASSWORD: str
    DB_USERNAME: str
    DB_HOST: str
    DB_PORT: str
    SQLALCHEMY_ECHO: bool = False

    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+aiomysql://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_DATABASE}"

    class Config:
        env_file: str = ".env"
        case_sensitive = True


# 설정 인스턴스 생성
settings = Settings()
