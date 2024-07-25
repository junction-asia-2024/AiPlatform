from pydantic import BaseModel, Field


class TodoItemBase(BaseModel):
    title: str 
    description: str | None = Field(default=None, description="빨래하러가기")
    completed: bool = Field(default=False)
    
    
class TodoItemCreate(TodoItemBase):
    pass

class TodoItemUpdate(TodoItemBase):
    id: int


class TodoItemInDBBase(TodoItemBase):
    id: int
    
    class Config:
        orm_mode: bool = True

class TodoItem(TodoItemInDBBase):
    pass

class TodoItemInDB(TodoItemInDBBase):
    pass