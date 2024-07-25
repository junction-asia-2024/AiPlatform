from sqlalchemy import Column, Integer, String, Boolean
from app.db.session import Base


class TodoItem(Base):
    __tablename__: str = "todo_items"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True, nullable=True)
    completed = Column(Boolean, default=False)