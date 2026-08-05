from pydantic import BaseModel, Field, ConfigDict
from pydantic.dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Cache:
    model: Any
    vocab: dict[int, str]
    allowfunc: list[str]
    functions: list[dict[str, Any]]
    param_types: dict[str, dict[str, Any]]
    mask: np.ndarray[Any, Any]
    numbers_mask: np.ndarray[Any, Any]
    no_comma: np.ndarray[Any, Any]
    mini_dict: list[tuple[int, str]]
    clean_dict: list[tuple[int, str]]


class FuncCallRes(BaseModel):
    prompt: str = Field(...)
    name: str = Field(...)
    parametor: dict[str, Any] = Field(...)


class FuncDefanition(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    parameters: dict[str, Any] = Field(...)
    returns: dict[str, Any] = Field(...)
