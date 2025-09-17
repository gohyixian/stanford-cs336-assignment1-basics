from collections import defaultdict
from collections.abc import Generator
from pathlib import Path

import regex as re

SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | Path, vocab_size: int, special_tokens: list[str], test_string: str = ""
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Method to train a BPE tokenizer on a given input text file.

    Args:
        input_path (str | os.Path): Path to a text file with BPE tokenizer training data.
        vocab_size (int): A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens (list[str]): A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.

    Returns:
        tuple: A tuple containing:
        ```
        - vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        - merges (list[tuple[bytes, bytes]]): A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
        ```
    """
    if test_string:
        corpus = test_string
    else:
        # 1. Read the input file and get the text data
        with open(input_path, encoding="utf-8") as f:
            corpus = f.read()

    # 2. Initialize the vocabulary
    ## Add in the special tokens first
    init_vocab_list = special_tokens + list(range(256))
    vocab = {
        idx: s_token.encode("utf-8") if idx == 0 else bytes([idx - len(special_tokens)])
        for idx, s_token in enumerate(init_vocab_list)
    }
    merges = []

    # 3. Pretokenization
    splitted_corpus = pretokenizer(corpus, SPLIT_PATTERN)
    utf8_encoded = [[bytes([b]) for b in t.encode("utf-8")] for t in splitted_corpus]
    iteration = 1
    while len(vocab) < vocab_size:
        freq_table = defaultdict(int)
        # 4. Find the most occuring pair
        for encoded_text in utf8_encoded:
            for idx in range(len(encoded_text) - 1):
                if len(encoded_text) > 1:
                    freq_table[(encoded_text[idx], encoded_text[idx + 1])] += 1

                else:
                    freq_table[encoded_text[idx]] += 1

        if not freq_table:
            break
        max_freq = max(freq_table.values())
        highest_freq_token_pair = [token_pair for token_pair, freq in freq_table.items() if freq == max_freq]

        ## If tie, prefer greater lexiography order
        if len(highest_freq_token_pair) > 1:
            selected_token_pair = max(highest_freq_token_pair)
        elif len(highest_freq_token_pair) > 0:
            selected_token_pair = highest_freq_token_pair[0]

        print(len(vocab) - 1, selected_token_pair)
        
        # 5. Merge the most occuring pair
        merges.append(selected_token_pair)
        vocab[len(vocab)] = selected_token_pair[0] + selected_token_pair[1]

        # 6. Update the corpus (use the selected_token_pair at the moment)
        tmp_corpus = []
        for encoded_text in utf8_encoded:
            block = []
            idx = 0
            while idx < len(encoded_text):
                if idx < len(encoded_text) - 1 and (encoded_text[idx], encoded_text[idx + 1]) == selected_token_pair:
                    block.append(vocab[len(vocab) - 1])
                    idx += 2
                else:
                    block.append(encoded_text[idx])
                    idx += 1
            tmp_corpus.append(block)
        utf8_encoded = tmp_corpus
        iteration += 1

    return vocab, merges


def pretokenizer(text: str, pattern: re.Pattern) -> Generator[None, None, str]:
    pattern = re.compile(pattern)
    for match in re.finditer(pattern, text):
        yield match.group(0)


if __name__ == "__main__":
    test_string = (
        """low low low! low low lower? lower widest. widest widest newest newest newest newest newest newest"""
    )

    vocab, merges = train_bpe("", 500, [], test_string=test_string)

    print(len(vocab))

    # print({token_id: repr(byte_val) for token_id, byte_val in vocab.items()})

    print("----------------")

    print(merges)
    for merge in merges:
        print(merge[0].decode("utf-8"), merge[1].decode("utf-8"))