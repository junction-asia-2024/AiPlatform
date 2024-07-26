from sqlalchemy import Column, Integer, String, Boolean
from app.db.session import Base


class TodoItem(Base):
    __tablename__ = "todo_items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True)
    description = Column(String(1000), index=True, nullable=True)
    completed = Column(Boolean, default=False)
