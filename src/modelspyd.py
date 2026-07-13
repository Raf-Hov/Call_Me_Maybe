from pydantic import BaseModel, Field
from typing import Optional, Any


class ParametModel(BaseModel):
    typ: Optional[str]


class ReturnModel(BaseModel):
    typ: Optional[str]


class FunctionDefinition(BaseModel):
    name: str
    descrit: str
    parameters: dict[str, ParametModel]
    retur: ReturnModel


class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
