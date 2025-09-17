import pickle
from cs336_basics.tokenization.bpe import train_bpe
# from cs336_basics.tokenization.weixin import train_bpe



# FILE = "tests/fixtures/corpus.en"
# VOCAB_SIZE = 500
# SPECIAL_TOKENS = [
#     "<|endoftext|>"
# ]
# vocab, merges = train_bpe(FILE, VOCAB_SIZE, SPECIAL_TOKENS)

# [print(k, ":", repr(v)) for k,v in vocab.items()]


FILE = "tests/fixtures/tinystories_sample_5M.txt"
VOCAB_SIZE = 1000
SPECIAL_TOKENS = [
    "<|endoftext|>"
]
vocab, merges = train_bpe(FILE, VOCAB_SIZE, SPECIAL_TOKENS)

# [print(k, ":", repr(v)) for k,v in vocab.items()]
[print(f"{i:>5}  {m}") for i,m in enumerate(merges)]


snapshot_path = "tests/_snapshots/test_train_bpe_special_tokens.pkl"
with open(snapshot_path, "rb") as f:
    snapshot = pickle.load(f)

print("\n\n\n================ (wrong_additional)")
wrong_additional = [i for i in merges if i not in snapshot["merges"]]
[print(i) for i in wrong_additional]

print("\n\n\n================ (not_found)")
not_found = [i for i in snapshot["merges"] if i not in merges]
[print(i) for i in not_found]
