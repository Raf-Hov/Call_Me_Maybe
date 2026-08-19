from .esiminch import ParsePyd
from .constrjson import JSONconstr
from typing import Any
from pydantic import ValidationError
from pathlib import Path
from .pydvalider import FuncCallRes
from datetime import datetime
import sys
import json


def main() -> None:
    start_time = datetime.now()
    parse = ParsePyd()
    jsn = JSONconstr(parse.cache)
    final_list: list[dict[str, Any]] = []

    for entry in parse.my_model.prompt_list:
        prompt_text = entry.get("prompt") if isinstance(entry, dict) else entry
        if not prompt_text:
            continue
        raw_json = jsn.generate_json(prompt_text)
        print()
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            print(
                f"Skipping prompt (model returned invalid JSON):"
                f"{prompt_text!r}\n  -> {raw_json!r}",
                file=sys.stderr,
            )
            continue
        try:
            result = FuncCallRes(
                prompt=prompt_text,
                name=parsed.get("name", ""),
                parameters=parsed.get("parameters", {}),
            )
        except ValidationError as e:
            print(
                f"Skipping the prompt (result failed validation): "
                f"{prompt_text!r}\n{e}",
                file=sys.stderr,
            )
            continue
        final_list.append(result.model_dump())
    output_path = Path(parse.my_model.args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)
    print(f"Written {len(final_list)} result(s) in {output_path}")
    elapsed_time = datetime.now() - start_time
    total_seconds = elapsed_time.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    print(f"\nAll done in {minutes}m {seconds}s!")


if __name__ == "__main__":
    main()
