from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


from app.core.config import settings


SQL_DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(SQL_DATABASE_URL, echo=True)
session_local = sessionmaker(
    autoflush=False, 
    autocommit=False,
    bind=engine,
    class_=AsyncSession
)

Base = declarative_base()


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    import asyncio
    asyncio.run(init_db())