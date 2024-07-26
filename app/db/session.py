from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
from app.core.config import settings

# 로깅 구성
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 설정에서 데이터베이스 URL 가져오기
SQL_DATABASE_URL = settings.DATABASE_URL

# 비동기 엔진 생성
engine = create_async_engine(SQL_DATABASE_URL, echo=settings.SQLALCHEMY_ECHO)

# 세션 팩토리 설정
async_session_local = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

# ORM 모델을 위한 베이스 클래스
Base = declarative_base()


async def init_db() -> None:
    """
    데이터베이스를 초기화하여 모든 테이블을 생성합니다.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("데이터베이스가 성공적으로 초기화되었습니다.")
    except Exception as e:
        logger.error(f"데이터베이스 초기화 중 오류 발생: {e}")


if __name__ == "__main__":
    import asyncio

    # 이벤트 루프에서 init_db 함수를 실행
    asyncio.run(init_db())
