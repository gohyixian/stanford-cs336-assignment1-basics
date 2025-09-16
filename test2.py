import json
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"

# Compare the learned merges to the expected output merges
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

with open(reference_vocab_path, encoding="utf-8") as f:
    gpt2_reference_vocab = json.load(f)
    reference_vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
    }

[print(k, ":", repr(v)) for k,v in reference_vocab.items()]
