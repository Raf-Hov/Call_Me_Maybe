from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Cache:
    model: Any  # im llm-na Tokenizatorov obekty
    logits_size: int  # logitneri size aveli chisht smakaneri hamar
    vocab: dict[int, str]  # henc vocabnna
    allowfunc: list[str]  # menak funkcianneri anunnern en
    functions: list[dict[str, Any]]  # henc functions_definition filena (dict)
    func_params: dict[str, int]
    param_types: dict[str, dict[str, Any]]
    mask: np.ndarray[Any, Any]
    numbers_mask: np.ndarray[Any, Any]
    no_comma: np.ndarray[Any, Any]
    mini_dict: list[tuple[int, str]]  # sax tokennern en voronq parunakum en
    # allowfnc yev prefixnery
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
