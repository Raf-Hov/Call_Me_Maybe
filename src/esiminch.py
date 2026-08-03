from .tokenizator import Tokenizer
from .pydvalider import FuncDefanition
from typing import Any
from pydantic import ValidationError
from enum import Enum
import sys
import numpy as np


class Phrases(str, Enum):
    NAME = '{"name":"'
    PARAM = '","parameters":{'
    PAKOX = '}'


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

    def mask_creator(self) -> None:
        dummy_ids = self.encode("dummy")
        self.dummys_logits_size = len(
            self.model.get_logits_from_input_ids(dummy_ids))
        mask = np.zeros(self.dummys_logits_size, dtype=bool)
        mask_no_comma = mask.copy()
        for k, s in self.clean_tok:
            if ',' in s:
                mask_no_comma[k] = False
        nums = np.zeros(self.dummys_logits_size, dtype=bool)
        print(nums)
        digi = set("0123456789.-, }")
        for i, d in self.clean_tok:
            if all(char in digi for char in d) or d == "null":
                nums[i] = True
        target = self.allowfnc
        for c in ['{"name":"', '","parameters":{', '}']:
            target.append(c)
        self.my_dict: list[tuple[int, str]] = []
        for a, j in self.clean_tok:
            if any(j in ph for ph in target):
                self.my_dict.append((a, j))
