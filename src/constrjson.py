from typing import Any
from string import printable
import json
import numpy as np
import re


class JSONconstr:
    def __init__(self):
        self.prefix = '{"name":"'
        self.bridge = ',"parameters":{'
        self.current_str = ""

    def json_gen(self, prompt: str, cache: Any) -> str:
        optimiz = []
        for i in cache.functions:
            optimiz.append({
                "name": i["name"],
                "description": i.get("description", ""),
                "parameters": i.get("parameters", {})
            })
        schema_hints = json.dumps(optimiz, separators=(',', ':'))
        my_own_propts = (
            f"System: You are a strict API. Output ONLY valid JSON matching "
            f"these schemas: {schema_hints}\n"
            r"Rule: For the regex field, NEVER output literal matches. "
            r"Always use proper regex sets "
            r"(e.g. '[aeiouAEIOU]', '[0-9]+', '\\bword\\b'). "
            r"For replacement, if asked for a character (e.g. asterisks), "
            r"output EXACTLY ONE character (e.g. '*')."
            "\n"
            f"User: {prompt}\n"
            "Tool Call: "
        )
        ids = cache.model.encode(prompt)
        max_token = 150
        bridge = False
        while (not self.current_str.replace(" ", "").replace("\n", "").endswith('}}')
                and len(ids) < len(prompt) + max_token):
            if self.prefix in self.current_str and self.bridge not in self.current_str:
                af_prefix = self.current_str.split(self.prefix)[1]
                posnam = [n for n in cache.allowfunc if n.startwith(af_prefix)]
                if len(posnam) == 1 and posnam[0] != af_prefix:
                    remaider = posnam[0][len(af_prefix):] + '"'
                    self.current_str += remaider
                    ids.extend(cache.model.encode(remaider))
                    continue
            if (self.current_str.endswith('"')
                and not bridge
                and self.prefix in self.current_str


