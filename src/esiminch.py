from .tokenizator import Tokenizer
from .pydvalider import FuncDefanition, Cache
from typing import Any
from pydantic import ValidationError
from enum import Enum
import sys
import numpy as np


class Phrases(str, Enum):
    NAME = '{"name":"'
    PARAM = '","parameters":{'
    PAKOX = '}'


class ParsePyd:
    def __init__(self) -> None:
        self.my_model = Tokenizer()
        self.allowfnc: list[str] = []
        self.par_type: dict[str, dict[str, Any]] = {}
        self.func_par: dict[str, int] = {}
        for func in self.my_model.funtions_list:
            try:
                funct = FuncDefanition(**func)
                self.allowfnc.append(funct.name)
                self.func_par[funct.name] = len(funct.parameters)
                par = func.get("parameters", {})
                self.par_type[funct.name] = {}
                for p, s in par.items():
                    self.par_type[funct.name][p] = s.get("type")
            except ValidationError as e:
                print(
                    "Validation ERROR: input data is invalid or incomplete.",
                    file=sys.stderr)
                print(e)
                sys.exit(1)
        self.mask_creator()

    def mask_creator(self) -> None:
        dummy_ids = self.my_model.encode("dummy")
        self.dummys_logits_size = len(
            self.my_model.llm.get_logits_from_input_ids(dummy_ids))
        mask = np.zeros(self.dummys_logits_size, dtype=bool)
        mask_no_comma = mask.copy()
        for k, s in self.my_model.clean_tok:
            if ',' in s:
                mask_no_comma[k] = False
        nums = np.zeros(self.dummys_logits_size, dtype=bool)
        digi = set("0123456789.-, }")
        for i, d in self.my_model.clean_tok:
            if all(char in digi for char in d) or d == "null":
                nums[i] = True
        target = self.allowfnc
        for c in ['{"name":"', '","parameters":{', '}']:
            target.append(c)
        my_dict: list[tuple[int, str]] = []
        for a, j in self.my_model.clean_tok:
            if any(j in ph for ph in target):
                my_dict.append((a, j))
        self.cache = Cache(
            model=self.my_model,
            vocab=self.my_model.vocab_dict,
            allowfunc=self.allowfnc,
            func_params=self.func_par,
            functions=self.my_model.funtions_list,
            param_types=self.par_type,
            mask=mask,
            numbers_mask=nums,
            no_comma=mask_no_comma,
            mini_dict=my_dict,
            clean_dict=self.my_model.clean_tok
        )
