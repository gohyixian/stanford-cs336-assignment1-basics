"""
Pretokenization of text. 

We do this to avoid assigning different IDs for similar texts say with different punctuations 
(a by-product of BPE merging), i.e. "dogs" vs "dogs!"
"""


import regex as re


# regex-based pre-tokenizer (used by GPT-2; Radford et al., 2019) from github.com/openai/tiktoken/pull/234/files
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def yield_pretokens(text: str): 
    """Pretokenizes text using GPT-2 pretokenization pattern."""
    
    pattern = re.compile(PAT)
    for match in re.finditer(pattern, text):
        yield match.group(0)



if __name__ == "__main__":
    print(list(yield_pretokens("Hey how are you?")))
    # ['Hey', ' how', ' are', ' you', '?']