# --This takes all toml files in TOML_DIRECTORY_CLEANED and parses them.
# --Then it adds a key-value pair "question_embedding" = [embedding array of the question].
# --It then saves a new file in the directory TOML_DIRECTORY_EMBEDDED. The file will have an "embedded_"-prefix, 
# ----which is a copy of the old toml, but with the embeddings included in the file.
# --Important to keep the "embedded_"-prefix, since other functions use that as a filter,
# ----to choose which file to read.

import os
import time
import threading
from mistralai import Mistral
from tomlkit import parse, dumps
from tqdm import tqdm
from config import (
    MISTRAL_KEY, 
    TOML_DIRECTORY_CLEANED, 
    TOML_DIRECTORY_EMBEDDED
    )

client = Mistral(api_key=MISTRAL_KEY)
rate_lock = threading.Lock()
last_call_time = [0.0]  # use list for mutability across threads
RATE_LIMIT = 6  # calls per second
MIN_INTERVAL = 1.0 / RATE_LIMIT


def rate_limited_embedding(question):
    with rate_lock:
        now = time.time()
        wait_time = MIN_INTERVAL - (now - last_call_time[0])
        if wait_time > 0:
            time.sleep(wait_time)
        last_call_time[0] = time.time()
        return client.embeddings.create(model="mistral-embed", inputs=question).data[0].embedding


def add_embeddings_to_toml(toml_dir):
    toml_files = [f for f in os.listdir(toml_dir) if f.endswith(".toml")]
    
    for toml_file in tqdm(toml_files, desc="Processing TOML files"):
        full_path = os.path.join(TOML_DIRECTORY_CLEANED, toml_file)
        with open(full_path, "r", encoding="utf-8") as f:
            toml_file_edit = parse(f.read())
        for question in toml_file_edit["questions"]:
            question["question_embedding"] = rate_limited_embedding(question["question"])
        os.makedirs(TOML_DIRECTORY_EMBEDDED, exist_ok=True)
        out_path = os.path.join(TOML_DIRECTORY_EMBEDDED, f"embedded_{toml_file}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(dumps(toml_file_edit))

# --------------------------------------------------------------#
# -------Write new toml files with embeddings included----------#
# --------------------------------------------------------------#
add_embeddings_to_toml(TOML_DIRECTORY_CLEANED)