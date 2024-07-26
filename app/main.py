from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.v1.endpoints import todo
from app.db.session import init_db


async def startup_event() -> None:
    await init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시 초기화 작업
    await init_db()
    yield
    # 애플리케이션 종료 시 작업이 필요한 경우 여기에 추가


app = FastAPI(lifespan=lifespan)
app.include_router(todo.router, prefix="/todos", tags=["todos"])
