from pydantic import BaseModel, Field, ConfigDict
from pydantic.dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Cache:
    model: Any = Field(...)
    vocab: dict[int, str] = Field(...)
    allowfunc: list[str] = Field(...)
    functions: list[dict[str, Any]] = Field(...)
    func_params: dict[str, int] = Field(...)
    param_types: dict[str, dict[str, Any]] = Field(...)
    mask: np.ndarray[Any, Any] = Field(...)
    numbers_mask: np.ndarray[Any, Any] = Field(...)
    no_comma: np.ndarray[Any, Any] = Field(...)
    mini_dict: list[tuple[int, str]] = Field(...)
    clean_dict: list[tuple[int, str]] = Field(...)


class FuncCallRes(BaseModel):
    prompt: str = Field(...)
    name: str = Field(...)
    parametor: dict[str, Any] = Field(...)


class FuncDefanition(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    parameters: dict[str, Any] = Field(...)
    returns: dict[str, Any] = Field(...)
