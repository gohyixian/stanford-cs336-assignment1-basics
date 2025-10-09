"""Byte-Pair Encoding Tokenizer implementation."""

import os
import regex as re
from collections import Counter
from typing import Iterable, Iterator
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.lm.tokenization.pretokenization import yield_pretokens




def init_vocab_gpt2(
    special_tokens: list[str] = []
) -> tuple[tuple[dict[int, bytes]], int]:
    """Initialises a vocabulary dict in GPT-2 style."""
    
    # init vocab: all bytes
    vocab: dict[int, bytes] = {i: tok.encode("utf-8") for i,tok in enumerate(special_tokens)}  # {byte int repr : byte} mapping
    
    # visible ASCII '!'..'~' (33..126) - space (32) excluded to match the sample ordering
    new_int_id = len(special_tokens) + 0
    for code in range(33, 127):
        vocab[new_int_id] = bytes([code])
        new_int_id += 1
    
    def _add(b: int):
        """Local add util."""
        nonlocal new_int_id
        bb = bytes([b])
        if bb not in vocab.values(): # dedup
            vocab[new_int_id] = bb
            new_int_id += 1
    
    # latin-1 “printables” (exc. NBSP and soft hyphen):
    for b in range(0xA1, 0xAD):  # A1..AC
        _add(b)
    for b in range(0xAE, 0x100):  # AE..FF
        _add(b)
    
    # controls:
    for b in range(0x00, 0x20):  # 00..1F
        _add(b)
    
    # space + del:
    _add(0x20)  # space
    _add(0x7F)  # del
    
    # C1 controls:
    for b in range(0x80, 0xA0):  # 80..9F
        _add(b)
    
    # NBSP and soft hyphen:
    _add(0xA0)  # NBSP
    _add(0xAD)  # soft hyphen (last)
    
    return vocab, new_int_id



def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Trains a custom vocab via BPE in GPT-2 style."""
    
    assert os.path.exists(input_path), f"File not found: {input_path}"
    
    
    # ##########################################
    # ## Step 1: Initialise Vocab (all pairs) ##
    # ##########################################
    # vocab: dict[int, bytes] = {i: tok.encode("utf-8") for i,tok in enumerate(special_tokens)}  # {byte int repr : byte} mapping
    # for i in range(256):
    #     id = len(special_tokens) + i
    #     vocab[id] = bytes([i])
    # new_int_id = len(special_tokens) + 256  # starting ID for new tokens
    
    # ---------------------
    # Optional: init vocab in gpt-2 style (works too, but different vocab order - customised)
    vocab, new_int_id = init_vocab_gpt2(special_tokens)
    # ---------------------
    
    # as a track record for the merges done
    merges: list[tuple[bytes, bytes]] = []
    
    # for single use in step 1.
    byte_to_id = {v: k for k, v in vocab.items() if len(v) == 1}
    
    
    
    ############################################################
    ## Step 2: Collect pre-tokens & convert to byte sequences ##
    ############################################################
    
    pre_token_counts: Counter[tuple[int, ...]] = Counter()
    
    # Load & chunk file as per cs336 pytest implementation
    # ========================================================================
    with open(input_path, "rb") as f:
        # Read the whole file into memory
        chunk = f.read().decode("utf-8", errors="ignore")
        
        # Split by special tokens
        pattern = "|".join(map(re.escape, special_tokens))
        parts = re.split(pattern, chunk)
        
        # Run pre-tokenization on chunk and store the counts for each pre-token's byte ID sequence
        for p in parts:
            for tok in yield_pretokens(p):
                # pre_token_counts[tuple(tok.encode("utf-8"))] += 1  # token byte int repr sequences: tuple(10, 103, 34, ..) += 1
                bs = tok.encode("utf-8")
                ids = tuple(byte_to_id[bytes([b])] for b in bs)  # e.g., (IDof'd', IDof'o', ...)
                pre_token_counts[ids] += 1
    
    #     num_processes = 4
        
    #     # determine each chunk's start & end indices
    #     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
    #     # The following is a serial implementation, but you can parallelize this
    #     # by sending each start/end pair to a set of processes.
    #     for start, end in zip(boundaries[:-1], boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
    #         # # replace special tokens with "" in chunk, i.e. ["<|endoftext|>", "<|pad|>"]
    #         # pattern = "|".join(map(re.escape, special_tokens))
    #         # chunk = re.sub(pattern, "", chunk)
            
    #         # split by special tokens
    #         pattern = "|".join(map(re.escape, special_tokens))
    #         # Split the chunk by the pattern
    #         parts = re.split(pattern, chunk)
    # # ========================================================================
            
    #         # Run pre-tokenization on chunk and store the counts for each pre-token's byte ID sequence
    #         for p in parts:
    #             for tok in yield_pretokens(p):
    #                 # pre_token_counts[tuple(tok.encode("utf-8"))] += 1  # token byte int repr sequences: tuple(10, 103, 34, ..) += 1
    #                 bs = tok.encode("utf-8")
    #                 ids = tuple(byte_to_id[bytes([b])] for b in bs)  # e.g., (IDof'd', IDof'o', ...)
    #                 pre_token_counts[ids] += 1
    
    
    #####################################
    ## Step 3: Merge pairs iteratively ##
    #####################################
    
    # NOTE: one iteration merges one (a,b) pair, and adds one new item to the vocab
    while len(vocab) < vocab_size:
        
        
        # count freq of adjacent pairs across all sequences
        # =================================================
        adj_pair_counts: Counter[tuple[int,int]] = Counter()
        # loop through all token byte int repr sequences: tuple(10, 103, 34, ..)
        for seq, freq in pre_token_counts.items():
            # loop through all adjacent byte int repr pairs: {10,103} {103,34} {34,..} ...
            zipped = zip(seq, seq[1:])  # Pre-compute sequence lengths to avoid repeated len() calls
            for a, b in zipped:
                adj_pair_counts[(a, b)] += freq  # increment by freq due to pre-token/word level freq, not 1!
        
        if not adj_pair_counts:
            break
        
        
        # find the most common pair
        # =========================
        # NOTE: deterministically break ties in pairs with same frequency by preferring the lexicographically greater pair.
        # NOTE: measure greater pair on byte pairs `(vocab[p[0]], vocab[p[1]])`, NOT byte int pairs `p`
        pair = max(adj_pair_counts, key=lambda p: (adj_pair_counts[p], (vocab[p[0]], vocab[p[1]])), default=None)
        if pair is None: break
        a, b = pair
        
        
        # merge that pair as new token & assign new ID (ID -> bytes)
        # ==========================================================
        vocab[new_int_id] = vocab[a] + vocab[b] # combine bytes together as pair, i.e. b"\x03" + b"\x04" = b"\x03\x04"
        merges.append((vocab[a], vocab[b])) # append merge as bytes
        
        
        # replace (a,b) pairs to `new_int_id` in all sequences
        # ====================================================
        new_pre_token_counts: Counter[tuple[int, ...]] = Counter()
        for seq, freq in pre_token_counts.items():
            
            # from old sequence, form new sequence via merging (assign as new token int id), if a & b are adjacent
            # -----------------------------------------------------
            new_seq = []
            i = 0
            while i < len(seq):
                
                # if not last and match pair, merge
                if ( i < len(seq) - 1 ) and ( seq[i] == a and seq[i+1] == b ):
                    new_seq.append(new_int_id)
                    i += 2
                
                # else preserve
                else:
                    new_seq.append(seq[i])
                    i += 1
            # -----------------------------------------------------
            
            # assign back to dict, use += in case other merges also result in this new seq
            new_pre_token_counts[tuple(new_seq)] += freq
        
        # update counter as from above
        pre_token_counts = new_pre_token_counts
        
        new_int_id += 1
    
    
    return vocab, merges









class BPETokenizer:
    """BPE Tokenizer class."""
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        """
        Initialize the BPE Tokenizer.
        
        Args:
            vocab: Dictionary mapping token IDs to byte sequences
            merges: List of merge operations (byte pairs) learned during training
            special_tokens: Optional list of special tokens to add to vocabulary
        """
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.special_tokens = special_tokens or []
        
        # Create reverse mapping from bytes to token IDs
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        
        # Add special tokens if provided
        if self.special_tokens:
            for special_token in self.special_tokens:
                special_token_bytes = special_token.encode('utf-8')
                if special_token_bytes not in self.token_to_id:
                    new_id = max(self.vocab.keys()) + 1
                    self.vocab[new_id] = special_token_bytes
                    self.token_to_id[special_token_bytes] = new_id
    
    
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        # load vocab
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                idx_str, byte_str = line.strip().split("\t")
                idx = int(idx_str)
                byte_seq = bytes.fromhex(byte_str)  # convert hex string back to bytes
                vocab[idx] = byte_seq
        
        # load merges
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                byte1_str, byte2_str = line.strip().split("\t")
                byte1 = bytes.fromhex(byte1_str)
                byte2 = bytes.fromhex(byte2_str)
                merges.append((byte1, byte2))
        
        return cls(vocab, merges, special_tokens)
    
    
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        
        Step 1: Pre-tokenize. We first pre-tokenize the sequence and represent each pre-token as a sequence of
        UTF-8 bytes, just as we did in BPE training. We will be merging these bytes within each pre-token into
        vocabulary elements, handling each pre-token independently (no merges across pre-token boundaries).
        
        Step 2: Apply the merges. We then take the sequence of vocabulary element merges created during BPE
        training, and apply it to our pre-tokens in the same order of creation.
        """
        # Handle special tokens first - they should be treated as atomic units
        # Process special tokens in order of length (longest first) to handle overlapping cases
        if self.special_tokens:
            # Sort special tokens by length (longest first) to handle overlapping cases
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            
            # Split text by special tokens and process each part
            parts = [text]
            for special_token in sorted_special_tokens:
                new_parts = []
                for part in parts:
                    # Skip if this part is already a special token (from previous iteration)
                    if part in self.special_tokens:
                        new_parts.append(part)
                    elif special_token in part:
                        split_parts = part.split(special_token)
                        for i, split_part in enumerate(split_parts):
                            if split_part:  # Only add non-empty parts
                                new_parts.append(split_part)
                            if i < len(split_parts) - 1:  # Add special token between parts
                                new_parts.append(special_token)
                    else:
                        new_parts.append(part)
                parts = new_parts
        else:
            parts = [text]
        
        # Process each part
        token_ids = []
        for part in parts:
            # Check if this part is a special token
            if part in self.special_tokens:
                special_token_bytes = part.encode('utf-8')
                if special_token_bytes in self.token_to_id:
                    token_ids.append(self.token_to_id[special_token_bytes])
                continue
            
            # Step 1: Pre-tokenize the part
            pretokens = list(yield_pretokens(part))
            
            # Step 2: Apply BPE merges to each pre-token (no merges across pre-token boundaries)
            for pretoken in pretokens:
                # Convert pre-token to bytes
                pretoken_bytes = pretoken.encode('utf-8')
                
                # Start with individual bytes
                tokens = [bytes([b]) for b in pretoken_bytes]
                
                # Apply merges in the same order they were learned
                for merge_pair in self.merges:
                    new_tokens = []
                    i = 0
                    while i < len(tokens):
                        # Check if current token and next token form the merge pair
                        if (i < len(tokens) - 1) and \
                           (tokens[i] == merge_pair[0]) and \
                           (tokens[i + 1] == merge_pair[1]):
                            # Merge the pair
                            new_tokens.append(merge_pair[0] + merge_pair[1])
                            i += 2  # Skip the next token since we merged it
                        else:
                            # Keep the current token
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens
                
                # Convert final tokens to token IDs
                for token in tokens:
                    if token in self.token_to_id:
                        token_ids.append(self.token_to_id[token])
                    else:
                        # Handle unknown tokens by falling back to individual bytes
                        for byte_val in token:
                            byte_token = bytes([byte_val])
                            if byte_token in self.token_to_id:
                                token_ids.append(self.token_to_id[byte_token])
        
        return token_ids
    
    
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            # Process each text chunk and yield token IDs one by one
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
    
    
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decodes a list of token IDs back to text.
        
        To decode a sequence of integer token IDs back to raw text, we can simply look up each ID's corresponding
        entries in the vocabulary (a byte sequence), concatenate them together, and then decode the bytes to a
        Unicode string. Note that input IDs are not guaranteed to map to valid Unicode strings (since a user
        could input any sequence of integer IDs). In the case that the input token IDs do not produce a valid
        Unicode string, you should replace the malformed bytes with the official Unicode replacement character
        U+FFFD.
        
        The errors argument of bytes.decode controls how Unicode decoding errors are handled, and
        using errors='replace' will automatically replace malformed data with the replacement marker.
        """
        # Concatenate all byte sequences corresponding to the token IDs
        byte_sequence = b''
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_sequence += self.vocab[token_id]
            else:
                # Handle unknown token IDs by skipping them or using a fallback
                # For robustness, we'll skip unknown IDs
                continue
        
        # Decode bytes to string, handling malformed Unicode with replacement character
        try:
            return byte_sequence.decode('utf-8')
        except UnicodeDecodeError:
            # Use 'replace' to handle malformed bytes with U+FFFD replacement character
            return byte_sequence.decode('utf-8', errors='replace')