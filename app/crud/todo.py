from typing import Sequence

from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession


from app.models.todo import TodoItem
from app.schemas.todo import TodoItemCreate, TodoItemUpdate


async def get_todo_item(db: AsyncSession, todo_id: int) -> TodoItem | None:
    result = await db.execute(select(TodoItem).filter(TodoItem.id == todo_id))
    return result.scalar_one_or_none()


async def get_todo_items(db: AsyncSession, skip: int = 0, limit: int = 10) -> Sequence[TodoItem]:
    result = await db.execute(select(TodoItem).offset(skip).limit(limit))
    return result.scalars().all()


async def create_todo_item(db: AsyncSession, todo: TodoItemCreate) -> TodoItem:
    db_todo = TodoItem(**todo.model_dump())
    db.add(db_todo)
    
    await db.commit()
    await db.refresh(db_todo)
    return db_todo

async def update_todo_item(db: AsyncSession, todo: TodoItemUpdate) -> TodoItem | None:
    db_todo = await get_todo_item(db, todo.id)
    
    if not db_todo:
        return None
    
    for key, value in todo.model_dump().items():
        setattr(db_todo, key, value)
    
    await db.commit()
    await db.refresh(db_todo)
    return db_todo


async def delete_todo_item(db: AsyncSession, todo_id: int) -> TodoItem | None:
    db_todo = await get_todo_item(db, todo_id)
    if not db_todo:
        return None
    await db.delete(db_todo)
    await db.commit()
    return db_todo