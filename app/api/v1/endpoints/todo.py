from typing import Sequence

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.todo import TodoItem as SchemasTodoItem
from app.schemas.todo import TodoItemCreate as SchemasTodoItemCreate
from app.schemas.todo import TodoItemUpdate as SchemasTodoItemUpdate

from app.crud.todo import create_todo_item as crud_todo_item_create
from app.crud.todo import get_todo_item as crud_todo_item_get
from app.crud.todo import get_todo_items as crud_todo_items_get
from app.crud.todo import update_todo_item as crud_todo_items_update
from app.crud.todo import delete_todo_item as crud_delete_todo_item 


from app.db.session import session_local


router = APIRouter()

async def get_db():
    async with session_local as session:
        yield session


@router.post("/", response_model=SchemasTodoItem)
async def create_todo_item(todo: SchemasTodoItemCreate, db: AsyncSession = Depends(get_db)) -> SchemasTodoItem:
    return await crud_todo_item_create(db=db, todo=todo)


@router.get("/{todo_id}", response_model=SchemasTodoItem)
async def read_todo_item(todo_id: int, db: AsyncSession = Depends(get_db)) -> SchemasTodoItem:
    db_todo = await crud_todo_item_get(db=db, todo_id=todo_id)
    if db_todo is None:
        raise HTTPException(status_code=404, detail="Todo not Found")

    return db_todo


@router.get("/", response_model=list[SchemasTodoItem])
async def read_todo_items(skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)) -> Sequence[SchemasTodoItem]:
    todo = await crud_todo_items_get(db=db, skip=skip, limit=limit)
    return todo


@router.put("/{todo_id}", response_model=SchemasTodoItem)
async def update_todo_item(todo_id: int, todo: SchemasTodoItemUpdate, db: AsyncSession = Depends(get_db)):
    db_todo = await crud_todo_items_update(db=db, todo=todo)
    if db_todo is None:
        raise HTTPException(status_code=404, detail="Todo not Found")
    return db_todo


@router.delete("/{todo_id}", response_model=SchemasTodoItem)
async def delete_todo_item(todo_id: int, db: AsyncSession = Depends(get_db)):
    db_todo = await crud_delete_todo_item(db=db, todo_id=todo_id)
    if db_todo is None:
        raise HTTPException(status_code=404, detail="Todo not found")
    return db_todo