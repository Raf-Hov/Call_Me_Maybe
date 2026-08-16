from typing import Any
import string
import json
import numpy as np
import re


class JSONconstr:
    def __init__(self, cache: Any) -> None:
        self.cache = cache
        self.pref = '{"name":"'
        self.bridge = '","parameters":{'
        
    def _allow_chars(self, curr_str: str, allow_names: list[str]) -> list[str]:
        if len(curr_str) < len(self.pref):
            return [self.pref[len(curr_str):]]
        teil = curr_str[len(self.pref):]
        if '"' not in teil:
            return [n[len(teil):] + '"'
                    for n in allow_names if n.startswith(teil)]
        funct = teil.split('"')[0]
        torg = self.pref + funct + self.bridge
        if len(curr_str) < len(torg):
            return [torg[len(curr_str):]]
        return list(string.printable)

    def generate_json(self, prompt_text: str) -> str:
        optimized_schemas = []
        for f in self.cache.functions:
            optimized_schemas.append(
                {
                    "name": f.get("name"),
                    "description": f.get("description", ""),
                    "parameters": f.get("parameters", {})
                }
            )
        schema_hints = json.dumps(optimized_schemas, separators=(',', ':'))
        prompt = (
            f"System: You are a strict API. Output ONLY valid JSON matching "
            f"these schemas: {schema_hints}\n"
            r"Rule: For the regex field, NEVER output literal matches. "
            r"Always use proper regex sets "
            r"(e.g. '[aeiouAEIOU]', '[0-9]+', '\\bword\\b'). "
            r"For replacement, if asked for a character (e.g. asterisks), "
            r"output EXACTLY ONE character (e.g. '*')."
            "\n"
            f"User: {prompt_text}\n"
            "Tool Call: "
        )
        self.current_str = self.pref
        self.input_ids = self.cache.model.encode(prompt)
        self.input_ids.extend(self.cache.model.encode(self.pref))
        self.bridge_dnel = False
        max_tokens = 150
        _toekn_count = len(self.input_ids)
        while (
            not
            self.current_str.replace(" ", "").replace("\n", "").endswith('}}')
                and len(self.input_ids) < _toekn_count + max_tokens):
            if (self.pref in self.current_str
                    and '","parameters":{' not in self.current_str):
                after_prefix = self.current_str.split(self.pref)[1]
                possible_names = [
                    n
                    for n in self.cache.allowfunc if n.startswith(after_prefix)]
                if (len(possible_names) == 1
                        and possible_names[0] != after_prefix):
                    remainder = possible_names[0][len(after_prefix):] + '"'
                    self.current_str += remainder
                    self.input_ids.extend(self.cache.model.encode(remainder))
                    continue
            if (self.current_str.endswith('"') and not self.bridge_dnel
                and self.pref in self.current_str
                    and len(self.current_str) > len(self.pref)):
                self.current_str += self.bridge
                self.input_ids.extend(self.cache.model.encode(self.bridge))
                self.bridge_dnel = True
                f_name = self.current_str.split(self.pref)[1].split('"')[0]
                if self.cache.func_params.get(f_name, 99) == 0:
                    self.current_str += "}}"
                    break
                active_schema = next(
                    (f for f in self.cache.functions if f["name"] == f_name),
                    None
                )
                if active_schema:
                    tiny_schema = json.dumps(
                        [{
                            "name": active_schema["name"],
                            "description": active_schema.get("description", ""),
                            "parameters": active_schema.get("parameters", {})
                        }],
                        separators=(',', ':')
                    )
                    tiny_prompt = (
                        f"System: Output valid JSON matching this schema: "
                        f"{tiny_schema}\n"
                        r"Rule: For the regex field, NEVER output "
                        r"literal matches. "
                        r'Always use proper regex sets '
                        r'(e.g. "[aeiouAEIOU]", "[0-9]+", "\\bword\\b"). '
                        r"For replacement, if asked for a character "
                        r"(e.g. asterisks), output EXACTLY ONE character "
                        r"(e.g. '*')."
                        "\n"
                        f"User: {prompt_text}\n"
                        f"Tool Call: {self.current_str}"
                    )

                    self.input_ids = self.cache.model.encode(tiny_prompt)
                else:
                    self.input_ids.extend(self.cache.model.encode(self.bridge))
                continue
            rules = self._allow_chars(self.current_str, self.cache.allowfunc)
            logits = np.array(
                self.cache.model.llm.get_logits_from_input_ids(self.input_ids))
            mask = np.zeros(self.cache.logits_size, dtype=bool)
            if len(rules) > 10:
                match = re.search(r'"name"\s*:\s*"([^"]+)', self.current_str)
                f_name = match.group(1) if match else ""
                params_str = self.current_str.split('"parameters"')[
                    1] if '"parameters"' in self.current_str else ""
                if params_str:
                    in_string = False
                    last_structural_colon = -1
                    last_structural_comma = -1
                    last_structural_brace = -1
                    for i, char in enumerate(params_str):
                        if char == '"':
                            if i == 0 or params_str[i-1] != '\\':
                                in_string = not in_string
                        elif not in_string:
                            if char == ':':
                                last_structural_colon = i
                            elif char == ',':
                                last_structural_comma = i
                            elif char == '}':
                                last_structural_brace = i
                    is_inside_value = (
                        last_structural_colon > last_structural_comma
                        and last_structural_colon > last_structural_brace
                    )
                    active_key = ""
                    if is_inside_value:
                        keys_found = re.findall(r'"([^"]+)"\s*:', params_str)
                        if keys_found:
                            active_key = keys_found[-1]
                    expected_type = self.cache.param_types.get(
                        f_name, {}).get(active_key, "Any")
                    param_count = len(re.findall(r'"([^"]+)"\s*:', params_str))
                    target_count = self.cache.func_params.get(f_name, 99)
                    if is_inside_value and expected_type == "number":
                        mask = self.cache.numbers_mask.copy()
                        if param_count == target_count:
                            for i, s in self.cache.clean_dict:
                                if ',' in s:
                                    mask[i] = False
                    elif is_inside_value and in_string:
                        mask = self.cache.mask.copy()
                        if param_count == target_count:
                            for i, s in self.cache.clean_dict:
                                if '",' in s.replace(" ", ""):
                                    mask[i] = False
                        if active_key == "regex":
                            for i, s in self.cache.clean_dict:
                                if ' ' in s:
                                    mask[i] = False
                            if re.search(r'"regex"\s*:\s*"$', params_str):
                                for i, s in self.cache.clean_dict:
                                    if not any(s.startswith(c)
                                               for c in ['[', '\\']):
                                        mask[i] = False
                    elif param_count == target_count:
                        clean_str = self.current_str.strip()
                        if clean_str.endswith('"'):
                            self.current_str = clean_str + "}}"
                            print(f"\rGenerating: {self.current_str}", end="",
                                  flush=True)
                            break
                        elif clean_str.endswith('}'):
                            self.current_str = clean_str + "}"
                            print(f"\rGenerating: {self.current_str}", end="",
                                  flush=True)
                            break
                        elif clean_str.endswith(','):
                            self.current_str = clean_str[:-1] + "}}"
                            print(f"\rGenerating: {self.current_str}", end="",
                                  flush=True)
                            break
                        else:
                            mask = self.cache.no_comma.copy()
                    else:
                        mask = self.cache.mask.copy()
                        is_expecting_key = (
                            params_str.strip().endswith('{')
                            or params_str.strip().endswith(',')
                        )
                        if is_expecting_key:
                            for i, s in self.cache.clean_dict:
                                cleaned = s.strip()
                                if not (cleaned.startswith('"') or
                                        not cleaned):
                                    mask[i] = False
            else:
                for i, s in self.cache.mini_dict:
                    if any(rule.startswith(s) for rule in rules):
                        mask[i] = True
            logits[~mask] = -np.inf
            best_id = int(np.argmax(logits))
            self.current_str += self.cache.vocab.get(best_id, "")
            self.input_ids.append(best_id)
            if (self.current_str.endswith('"')
                and not self.bridge_dnel
                    and self.pref in self.current_str):
                bridge = ',"parameters":{'
                self.current_str += bridge
                self.input_ids.extend(self.cache.model.encode(bridge))
                self.bridge_dnel = True
                continue
            else:
                print(f"\rGenerating: {self.current_str}", end="", flush=True)
        print()
        return self.current_str
