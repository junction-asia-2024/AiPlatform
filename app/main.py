from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.v1.endpoints import todo
from app.db.session import init_db

async def startup_event() -> None:
    await init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    
    

app = FastAPI(lifespan=lifespan)
app.include_router(todo.router, prefix="/todos", tags=["todos"])
