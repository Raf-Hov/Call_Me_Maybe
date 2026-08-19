from .aparse import ArgPars
from llm_sdk import Small_LLM_Model
from typing import Any
from string import printable


class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: dict[str, "_TrieNode"] = {}
        self.token_id: int | None = None


class VocabTrie:
    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.root = _TrieNode()
        for token, token_id in token_to_id.items():
            node = self.root
            for ch in token:
                node = node.children.setdefault(ch, _TrieNode())
            node.token_id = token_id

    def longest_match(self, text: str, start: int) -> "tuple[int, int] | None":
        node = self.root
        i = start
        best: "tuple[int, int] | None" = None
        while i < len(text) and text[i] in node.children:
            node = node.children[text[i]]
            i += 1
            if node.token_id is not None:
                best = (i, node.token_id)
        return best


class Tokenizer(ArgPars):
    def __init__(self) -> None:
        self.args = self.parse_argum()
        self.funtions_list: list[dict[str, Any]] = self.load_json_file(
                self.args.functions_definition
                )
        self.prompt_list: list[dict[str, Any]] = self.load_json_file(
                self.args.input
                )
        self.llm = Small_LLM_Model(model_name=self.args.model)
        vocab: dict[str, int] = self.load_json_file(
            self.llm.get_path_to_vocab_file())
        self.vocab_dict: dict[int, str] = {
                v: k.replace('Ġ', ' ') for k, v in vocab.items()}
        self.set_of_printabls: set[str] = set(printable)
        self.valid_id: list[int] = []
        self.clean_tok: list[tuple[int, str]] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        for token, id in vocab.items():
            self.token_to_id[token] = id
            self.id_to_token[id] = token
        for token_id, token_str in self.vocab_dict.items():
            if token_str and all(is_valid in self.set_of_printabls
                                 for is_valid in token_str):
                self.valid_id.append(token_id)
        for id, token in self.vocab_dict.items():
            if token and all(
                  valid in self.set_of_printabls for valid in token):
                self.clean_tok.append((id, token))
        self.max_token_len = max((len(k) for k in self.token_to_id.keys()),
                                 default=1)
        self.trie = VocabTrie(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        text = text.replace(" ", "Ġ")
        my_list: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            match = self.trie.longest_match(text, i)
            if match is not None:
                end, token_id = match
                my_list.append(token_id)
                i = end
            else:
                i += 1
        return my_list

    def decode(self, ids: list[int]) -> str:
        text = "".join([self.id_to_token.get(i, "") for i in ids])
        return text.replace("Ġ", " ")
