from .esiminch import ParsePyd
from .constrjson import JSONconstr
from .tokenizator import Tokenizer
from typing import Any
from pydantic import ValidationError
from pathlib import Path
from .pydvalider import FuncCallRes
import sys
import json


def main() -> None:
    parse = ParsePyd()
    jsn = JSONconstr()
    tok = Tokenizer()
    final_list: list[dict[str, Any]] = []
    for prompt in tok.prompt_list:
        prompt_txt: str = prompt['prompt']
        print(f"\nPrompt: {prompt_txt}")
        raw_json_string = jsn.generate_constrained_json(
            prompt_txt, parse.cache)
        try:
            json_str = raw_json_string
            extracted_dict = json.loads(json_str)

            fn_name = extracted_dict.get("name")
            expected_params = {}
            for fn in tok.funtions_list:
                if fn.get("name") == fn_name:
                    expected_params = fn.get("parameters", {})
                    break
            if "parameters" in extracted_dict:
                for key, val in extracted_dict["parameters"].items():
                    if (
                        key in expected_params
                        and expected_params[key].get("type") == "number"
                    ):
                        if isinstance(val, int) and not isinstance(val, bool):
                            extracted_dict["parameters"][key] = float(val)
                    elif isinstance(val, str):
                        extracted_dict["parameters"][key] = val.strip()

            final_data = {
                "prompt": prompt_txt,
                "name": extracted_dict["name"],
                "parameters": extracted_dict["parameters"]
            }
            result = FuncCallRes(**final_data)
            final_list.append(result.model_dump())
            print()
        except ValidationError as e:
            print(
                "Validation failed: output data is invalid or incomplete.",
                file=sys.stderr)
            print(e.errors()[0])
            sys.exit(1)
        except Exception as e:
            print(
                f"An unexpected error occured.\nDetails: {e}", file=sys.stderr)
            sys.exit(1)

    output_file = Path(tok.args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    Path(tok.args.output).touch(exist_ok=True)
    if not output_file.is_file():
        print(
            f"This file '{tok.args.output}' "
            "is not found, or it is a directory.",
            file=sys.stderr)
        sys.exit(1)
    try:
        with open(tok.args.output, 'w') as file:
            json.dump(final_list, file, indent=4)
    except PermissionError:
        print(f"Permission denied in this file {tok.args.output}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
