from typing import Any
from string import printable
import json
import numpy as np
import re


class JSONconstr:
    def allow_char(curr_str: str, names: list[str]) -> list[str]:
        prefix = '{"name":"'
        after_prefix = curr_str[len(prefix):]
        func_name = after_prefix.split('"')[0]
        target = prefix + func_name + '","parameters":{'
        if len(curr_str) < len(prefix):
            return [prefix[len(curr_str)]]
        if '"' not in after_prefix:
            return [name[len(after_prefix):] + '"'
                    for name in names if name.startswith(after_prefix)]
        if len(curr_str) < len(target):
            return [target[len(curr_str):]]
        return list(printable)

    def json_gen(self, prompt: str, cache: Any) -> str:
        optimiz = []
        for i in cache.functions:
            optimiz.append({
                "name": i["name"],
                "description": i.get("description", ""),
                "parameters": i.get("parameters", {})
            })
        schema_hints = json.dumps(optimiz, separators=(',', ':'))
        print(schema_hints)
