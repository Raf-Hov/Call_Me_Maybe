from pydantic import BaseModel, Field
from typing import Optional, Any

__all__ = ["ParametModel",
           "ReturnModel",
           "FunctionDefinition",
           "FunctionCallOutput"]


class ParametModel(BaseModel):
    type: Optional[str]


class ReturnModel(BaseModel):
    type: Optional[str]


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParametModel]
    returns: ReturnModel


class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
