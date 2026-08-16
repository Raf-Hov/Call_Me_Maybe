import argparse as ap
import os
from typing import Any
import json
import sys


class ArgPars:
    def parse_argum(self) -> ap.Namespace:
        parse = ap.ArgumentParser(
            prog="CallMeMaybe",
            usage="uv run python -m src [--functions_definition "
                  "<function_definition_file>] [--input <input_file>] "
                  "[--output <output_file>]"
        )
        parse.add_argument(
            '--functions_definition',
            type=str,
            default="data/input/functions_definition.json",
            help="Path to JSON file containing the functions definitions."
        )
        parse.add_argument(
            '--input',
            type=str,
            default="data/input/function_calling_tests.json",
            help="Path to the file containing the prompts."
        )
        parse.add_argument(
            '--output',
            type=str,
            default="data/output/function_calling_results.json",
            help="Path to the JSON output file."
        )
        parse.add_argument(
            '--model',
            type=str,
            default="Qwen/Qwen3-0.6B",
            help="HuggingFace Model ID"
        )
        args = parse.parse_args()
        return args

    def load_json_file(self, file_path: str) -> Any:
        if not os.path.isfile(file_path):
            raise SystemExit(
                f"This file {file_path} is not found, or it is a directory.")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("JSON Decode Error: The JSON file is invalid or corrupted. "
                  "A required key or value is missing.", file=sys.stderr)
            sys.exit(1)
        except PermissionError:
            print(f"Permission denied in this file {file_path}",
                  file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"detected ERROR: \n{e}", file=sys.stderr)
            sys.exit(1)
