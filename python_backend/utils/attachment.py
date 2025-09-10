from pydantic import BaseModel
from typing import Optional

class Attachment(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    content: Optional[str] = None
