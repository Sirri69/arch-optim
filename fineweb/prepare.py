"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
from multiprocessing import freeze_support

from datatrove.pipeline.readers import ParquetReader

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample/10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# limit determines how many documents will be streamed (remove for all)
fw = ParquetReader(f"hf://datasets/HuggingFaceFW/fineweb-edu/{remote_name}", limit=5_000_000_000)
# for document in fw():
#     # do something with document
#     print(document)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    try:
        # Clean/normalize the text first
        text = doc.text.encode('utf-8', errors='ignore').decode('utf-8')
        tokens = [eot]
        tokens.extend(enc.encode_ordinary(text))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16
    except Exception as e:
        print(f"Warning: Error processing document: {e}")
        return np.array([eot], dtype=np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

if __name__ == '__main__':
    freeze_support()
    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        split = "val" if shard_index == 0 else "train"
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        # if os.path.exists(f"{DATA_CACHE_DIR}/edufineweb_{split}_{shard_index:06d}.npy"):
        #     print("Skipping existing shards")
            
            
        for tokens in pool.imap(tokenize, fw(), chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                print(f"Checking if {filename}.npy exists")
                if not os.path.exists(filename + '.npy'):
                    print(f"Writing {filename}.npy")
                    # split the document into whatever fits in this shard; the remainder goes to next one
                    remainder = shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                    write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])