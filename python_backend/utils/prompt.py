from pydantic import BaseModel
from typing import List, Union

class ContentPart(BaseModel):
    type: str
    text: str

class ClientMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]
