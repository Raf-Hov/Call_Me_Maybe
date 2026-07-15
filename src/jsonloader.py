import argparse as ap
import json
import sys
from typing import Any


class ArgPars:
    def arg_parser(self) -> ap.Namespace:
        parse = ap.ArgumentParser(description="Call Me Maybe: Function calling")
        parse.add_argument(
            "--functions_definition",
            type=str,
            default="data/input/functions_definition.json",
            help="Path to the functions_definition JSON file"
        )
        parse.add_argument(
            "--input",
            type=str,
            default="data/input/function_calling_tests.json",
            help="Path to the input test JSON file"
        )
        parse.add_argument(
            "--output",
            type=str,
            default="data/output/function_calls.json",
            help="Path to the output JSON file"
        )
        return parse.parse_args()

    def load_json_file(self, path_to_file: str) -> Any:
        try:
            with open(path_to_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except OSError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: {e}", file=sys.stderr)

    def container(self) -> None:
        arg = self.arg_parser()
        self.funcall = self.load_json_file(arg.functions_definition)
        self.tokens = self.load_json_file(arg.input)

