from .tokenizator import Tokenizer
from .pydvalider import FuncDefanition
from typing import Any
from pydantic import ValidationError
import sys


class ParsePyd(Tokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.allowfnc: list[str] = []
        par_type: dict[str, dict[str, Any]] = {}
        for func in self.funtions_list:
            try:
                funct = FuncDefanition(**func)
                self.allowfnc.append(funct.name)
                par = func.get("parameters", {})
                par_type[funct.name] = {}
                for p, s in par.items():
                    par_type[funct.name][p] = s.get("type")
            except ValidationError as e:
                print("Validation ERROR: input data is invalid or incomplete.",
                      file=sys.stderr)
                print(e)
                sys.exit(1)
