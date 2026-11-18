import gzip

import numpy as np
import pytest

from models.sam3_reference import SimpleTokenizer
from models.sam3_pipeline import tokenize_prompt


@pytest.fixture
def tokenizer(tmp_path):
    path = tmp_path / "vocab.txt.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(b"#version: 0.2\nb a\nba g</w>\n")
    return SimpleTokenizer(path)


def test_context_capacity_counts_both_special_tokens(tokenizer):
    tokens = tokenize_prompt(tokenizer, " ".join(["a"] * 30))
    assert tokens.shape == (1, 32)
    assert tokens.dtype == np.int64
    assert tokens[0, 0] == tokenizer.sot_token_id
    assert tokens[0, -1] == tokenizer.eot_token_id
    assert np.all(tokens[0, 1:-1] == tokenizer.encoder["a</w>"])


def test_over_capacity_prompt_is_rejected_not_truncated(tokenizer):
    with pytest.raises(ValueError, match="32|context"):
        tokenize_prompt(tokenizer, " ".join(["a"] * 31))


def test_tokenizer_normalization_matches_reference_lower_cleaning(tokenizer):
    np.testing.assert_array_equal(tokenize_prompt(tokenizer, "  A &amp; B  "), tokenize_prompt(tokenizer, "a & b"))
