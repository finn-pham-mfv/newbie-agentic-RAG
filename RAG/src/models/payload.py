from pydantic import BaseModel


class Payload(BaseModel):
    content: str
    source: list[str]
