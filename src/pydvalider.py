from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from typing import Any


@dataclass
class Cache:
    model: Any
    vocab: dict[int, str]
    func_name: list[str]


class FuncCallRes(BaseModel):
    prompt: str
    name: str
    parametor: dict[str, Any]


class FuncDefanition(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    parameters: dict[str, Any] = Field(...)
    returns: dict[str, Any] = Field(...)
