from typing import Any
import string
import json
import numpy as np
import re


class JSONconstr:
    def __init__(self):
        self.prefix = '{"name":"'
        self.bridge = ',"parameters":{'
        self.current_str = ""

    def get_allowed_chars(
            self, current_str: str, allowed_names: list[str]) -> list[str]:
        if len(current_str) < len(self.prefix):
            return [self.prefix[len(current_str):]]

        after_prefix = current_str[len(self.prefix):]
        if '"' not in after_prefix:
            return [name[len(after_prefix):] + '"'
                    for name in allowed_names if name.startswith(after_prefix)]
        func_name = after_prefix.split('"')[0]
        target = self.prefix + func_name + '","parameters":{'
        if len(current_str) < len(target):
            return [target[len(current_str):]]

        return list(string.printable)

    def generate_constrained_json(
            self, prompt_text: str, cache: Any) -> str:
        optimized_schemas = []
        for f in cache.functions:
            optimized_schemas.append({
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": f.get("parameters", {})
            })
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
        input_ids = cache.model.encode(prompt)
        vocab_size = len(cache.model.llm.get_logits_from_input_ids(input_ids))
        current_str = ""
        prefix = '{"name":"'
        current_str = prefix
        input_ids.extend(cache.model.encode(prefix))
        bridge_injected = False
        max_tokens = 150
        while (
            not current_str.replace(" ", "").replace("\n", "").endswith('}}')
                and len(input_ids) < len(prompt) + max_tokens):
            if prefix in current_str and '","parameters":{' not in current_str:
                after_prefix = current_str.split(prefix)[1]
                possible_names = [
                    n for n in cache.allowfunc if n.startswith(after_prefix)]
                if (len(possible_names) == 1
                        and possible_names[0] != after_prefix):
                    remainder = possible_names[0][len(after_prefix):] + '"'
                    current_str += remainder
                    input_ids.extend(cache.model.encode(remainder))
                    continue
            if (current_str.endswith('"')
                and not bridge_injected
                and prefix in current_str
                    and len(current_str) > len(prefix)):
                bridge = ',"parameters":{'
                current_str += bridge
                input_ids.extend(cache.model.encode(bridge))
                bridge_injected = True
                func_name = current_str.split('"name":"')[1].split('"')[0]
                if cache.func_params.get(func_name, 99) == 0:
                    current_str += "}}"
                    break
                active_schema = next(
                    (f for f in cache.functions if f["name"] == func_name),
                    None
                )
                if active_schema:
                    tiny_schema = json.dumps(
                        [{
                            "name": active_schema["name"],
                            "description": active_schema.get(
                                "description", ""),
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
                        f"Tool Call: {current_str}"
                    )
                    input_ids = cache.model.encode(tiny_prompt)
                else:
                    input_ids.extend(cache.model.encode(bridge))
                continue
            rules = self.get_allowed_chars(current_str, cache.allowfunc)
            logits = np.array(
                cache.model.llm.get_logits_from_input_ids(input_ids))
            mask = np.zeros(vocab_size, dtype=bool)
            if len(rules) > 10:
                match = re.search(r'"name"\s*:\s*"([^"]+)', current_str)
                func_name = match.group(1) if match else ""
                params_str = current_str.split('"parameters"')[
                    1] if '"parameters"' in current_str else ""
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
                    expected_type = cache.param_types.get(
                        func_name, {}).get(active_key, "Any")
                    param_count = len(re.findall(r'"([^"]+)"\s*:', params_str))
                    target_count = cache.func_params.get(func_name, 99)
                    if is_inside_value and expected_type == "number":
                        mask = cache.numbers_mask.copy()
                        if param_count == target_count:
                            for i, s in cache.clean_dict_items:
                                if ',' in s:
                                    mask[i] = False
                    elif is_inside_value and in_string:
                        mask = cache.mask.copy()
                        if param_count == target_count:
                            for i, s in cache.clean_dict:
                                if '",' in s.replace(" ", ""):
                                    mask[i] = False
                        if active_key == "regex":
                            for i, s in cache.clean_dict:
                                if ' ' in s:
                                    mask[i] = False
                            if re.search(r'"regex"\s*:\s*"$', params_str):
                                for i, s in cache.clean_dict:
                                    if not any(s.startswith(c)
                                               for c in ['[', '\\']):
                                        mask[i] = False
                    elif param_count == target_count:
                        clean_str = current_str.strip()
                        if clean_str.endswith('"'):
                            current_str = clean_str + "}}"
                            print(f"\rGenerating: {current_str}",
                                  end="", flush=True)
                            break
                        elif clean_str.endswith('}'):
                            current_str = clean_str + "}"
                            print(f"\rGenerating: {current_str}",
                                  end="", flush=True)
                            break
                        elif clean_str.endswith(','):
                            current_str = clean_str[:-1] + "}}"
                            print(f"\rGenerating: {current_str}",
                                  end="", flush=True)
                            break
                        else:
                            mask = cache.no_comma.copy()
                    else:
                        mask = cache.mask.copy()
                        is_expecting_key = (
                            params_str.strip().endswith('{')
                            or params_str.strip().endswith(',')
                        )
                        if is_expecting_key:
                            for i, s in cache.clean_dict:
                                cleaned = s.strip()
                                if not (cleaned.startswith('"')
                                        or not cleaned):
                                    mask[i] = False
            else:
                for i, s in cache.mini_dict:
                    if any(rule.startswith(s) for rule in rules):
                        mask[i] = True
            logits[~mask] = -np.inf
            best_id = int(np.argmax(logits))
            current_str += cache.vocab.get(best_id, "")
            input_ids.append(best_id)
            if (current_str.endswith('"')
                    and not bridge_injected
                    and prefix in current_str):
                bridge = ',"parameters":{'
                current_str += bridge
                input_ids.extend(cache.model.encode(bridge))
                bridge_injected = True
                continue
            else:
                print(f"\rGenerating: {current_str}", end="", flush=True)
        print()
        return current_str
