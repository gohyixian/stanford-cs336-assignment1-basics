"""Byte-Pair Encoding Tokenizer implementation."""

import os
import regex as re
from collections import Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenization.pretokenization import yield_pretokens




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
        num_processes = 4
        
        # determine each chunk's start & end indices
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            # replace special tokens with "" in chunk, i.e. ["<|endoftext|>", "<|pad|>"]
            pattern = "|".join(map(re.escape, special_tokens))
            chunk = re.sub(pattern, "", chunk)
    # ========================================================================
            
            # Run pre-tokenization on chunk and store the counts for each pre-token's byte ID sequence
            for tok in yield_pretokens(chunk):
                # pre_token_counts[tuple(tok.encode("utf-8"))] += 1  # token byte int repr sequences: tuple(10, 103, 34, ..) += 1
                bs = tok.encode("utf-8")
                ids = tuple(byte_to_id[bytes([b])] for b in bs)  # e.g., (IDof'd', IDof'o', ...)
                pre_token_counts[ids] += 1
    
    
    
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
            for a, b in zip(seq, seq[1:]):
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