from cs336_basics.tokenization.bpe import train_bpe
# from cs336_basics.tokenization.weixin import train_bpe

# scalene test3.py


FILE = "tests/fixtures/corpus.en"
VOCAB_SIZE = 500
SPECIAL_TOKENS = [
    "<|endoftext|>"
]
vocab, merges = train_bpe(FILE, VOCAB_SIZE, SPECIAL_TOKENS)

# print(len(vocab))
[print(k, ":", repr(v)) for k,v in vocab.items()]
# print(len(merges))
# [print(i) for i in merges]
