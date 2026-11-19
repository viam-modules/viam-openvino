# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Adapted from SAM3 tokenizer_ve.py at 31cb7e624e4fe95301b1be1d4e809111593d2e45. Inherited SAM/CLIP/OpenCLIP notices are in third_party/.
"""The official lower-cleaning BPE tokenizer, restricted to text-only inference."""

from __future__ import annotations

import gzip
import html
from functools import lru_cache
from pathlib import Path

import ftfy
import regex as re


@lru_cache()
def bytes_to_unicode():
    values = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    characters = values[:]
    offset = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + offset)
            offset += 1
    return dict(zip(values, map(chr, characters)))


def _pairs(word):
    return set(zip(word, word[1:]))


class SimpleTokenizer:
    def __init__(self, bpe_path: Path):
        self.byte_encoder = bytes_to_unicode()
        with gzip.open(bpe_path, "rt", encoding="utf-8") as stream:
            merges = stream.read().split("\n")[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(self.byte_encoder.values())
        vocab += [value + "</w>" for value in vocab]
        vocab += ["".join(merge) for merge in merges]
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {token: token for token in special_tokens}
        self.sot_token_id = self.encoder[special_tokens[0]]
        self.eot_token_id = self.encoder[special_tokens[1]]
        self.pat = re.compile(
            "|".join(special_tokens) + r"|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            result = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    result.extend(word[index:])
                    break
                result.extend(word[index:next_index])
                index = next_index
                if index < len(word) - 1 and word[index] == first and word[index + 1] == second:
                    result.append(first + second)
                    index += 2
                else:
                    result.append(word[index])
                    index += 1
            word = tuple(result)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        encoded = " ".join(word)
        self.cache[token] = encoded
        return encoded

    def encode(self, text):
        text = html.unescape(html.unescape(ftfy.fix_text(text))).strip()
        text = re.sub(r"\s+", " ", text).strip().lower()
        result = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[value] for value in token.encode("utf-8"))
            result.extend(self.encoder[part] for part in self.bpe(token).split(" "))
        return result
