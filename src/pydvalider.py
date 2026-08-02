from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from typing import Any


@dataclass
class Cache:
    model: Any
    vocab: dict[int, str]
    allowfunc: list[str]
    functions: list[dict[str, Any]]
    param_types: dict[str, dict[str, Any]]


class FuncCallRes(BaseModel):
    prompt: str
    name: str
    parametor: dict[str, Any]


class FuncDefanition(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    parameters: dict[str, Any] = Field(...)
    returns: dict[str, Any] = Field(...)
