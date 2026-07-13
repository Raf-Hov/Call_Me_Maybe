from src.modelspyd import (FunctionCallOutput,
                           FunctionDefinition,
                           ReturnModel,
                           ParametModel)
import argparse as ap
import json
import sys


def parce_argp() -> ap.Namespace:
    parse = ap.ArgumentParser(description="Call me maybe: Function Calling")
    parse.add_argument(
        '--functions_definition',
        type=str,
        default='data/input/functions_definition.json',
        help='Path to the functions definition JSON file'
    )
    parse.add_argument(
        '--input',
        type=str,
        default='data/input/function_calling_tests.json',
        help='Path to the input tests JSON file'
    )
    parse.add_argument(
        '--output',
        type=str,
        default='data/output/function_calling_results.json',
        help='Path to the output results JSON file'
    )
    return parse.parse_args()


def load_json_file(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON.",
              file=sys.stderr)
        exit(1)


if __name__ == "__main__":
    args = parce_argp()
    
    # 1. Читаем сырые данные с диска
    raw_functions = load_json_file(args.functions_definition)
    raw_tests = load_json_file(args.input)
    
    # 2. Валидируем функции через Pydantic-модель
    # Мы ожидаем, что в raw_functions лежит список (array) функций
    validated_functions = []
    try:
        for item in raw_functions:
            # Создаем объект модели, распаковывая словарь из JSON
            func_def = FunctionDefinition(**item)
            validated_functions.append(func_def)
    except Exception as e: # Здесь pydantic.ValidationError, если структура битая
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3. Финальный проверочный принт
    print(f"Successfully loaded {len(validated_functions)} functions and {len(raw_tests)} tests.")
