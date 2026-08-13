from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Cache:
    model: Any
    vocab: dict[int, str]
    allowfunc: list[str]
    functions: list[dict[str, Any]]
    func_params: dict[str, int]
    param_types: dict[str, dict[str, Any]]
    mask: np.ndarray[Any, Any]
    numbers_mask: np.ndarray[Any, Any]
    no_comma: np.ndarray[Any, Any]
    mini_dict: list[tuple[int, str]]
    clean_dict: list[tuple[int, str]]


class FuncCallRes(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]


class FuncDefanition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]
