from .esiminch import ParsePyd
from .constrjson import JSONconstr
from typing import Any
from pydantic import ValidationError
from pathlib import Path
from .pydvalider import FuncCallRes
import sys
import json


def main() -> None:
    parse = ParsePyd()
    jsn = JSONconstr(parse.cache)
    final_list: list[dict[str, Any]] = []
    jsn.generate_json("What is the sum of 2 and 3?")



if __name__ == "__main__":
    main()
