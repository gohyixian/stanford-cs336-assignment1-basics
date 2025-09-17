import regex as re

special_tokens = [
    "<|endoftext|>",
    "<|pad|>"
]


chunk = "aoerghcieygfnxrf\n<|endoftext|>\naoufhcnaeifcgr\n<|pad|>\nifughcbadfigcbr"


pattern = "|".join(map(re.escape, special_tokens))
chunk = re.sub(pattern, "", chunk)

print(chunk)