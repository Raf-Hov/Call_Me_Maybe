from typing import Any
import string
from .esiminch import ParsePyd


def main() -> None:
    tok = ParsePyd()
    tok.mask_creator()


main()
