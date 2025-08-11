# --This script takes all pdf files in PDF_DIRECTORY (default=pdf_data) and parses them.
# --It also tokenizes, Chunks up text, creates embeddings.
# --Then it inserts the embeddings to chromadb, with metadata.
# --Only nedds to be run if you want to add documents to the db.
# --Check config.py, before running, to make sure you have the right settings for your case.

import fitz  # PyMuPDF
import os
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import multiprocessing
from ratelimit import limits, sleep_and_retry

from config import (
    PDF_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    get_collection,
    get_client
)
from helping_scripts.chunking import chunk_pdf_recursive_token_size


collection = get_collection() # set up db
client = get_client() # OpenAI client for embeddings


# -----------------------------------------------#
# --------------------Parse----------------------#
# -----------------------------------------------#
def parse_document(pdf_path, filename):
    doc = fitz.open(pdf_path)
    text_and_pagenumber = []  # List [(page_number, page_text)]

    for i, page in enumerate(doc):
        text = page.get_text(sort=True) # sort helps keep the right reading order in the page
        if text.strip():  # Skip empty pages
            text_and_pagenumber.append((i + 1, text + " "))
    doc.close()
    return text_and_pagenumber


# -----------------------------------------------#
# -----Embedd PDFs and Insert to ChromaDB--------#
# -----------------------------------------------#
MAX_CALLS_PER_SECOND = 6

@sleep_and_retry
@limits(calls=MAX_CALLS_PER_SECOND, period=1)
def call_embedding_api(model, inputs):
    return client.embeddings.create(model="mistral-embed", inputs=inputs)

# Get embeddings of chunks from client, store with metadata in db
def batch_embed_and_insert(chunks, batch_size=50):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]
        ids = [chunk["metadata"]["id"] for chunk in batch]

        try:
            response = call_embedding_api(EMBEDDING_MODEL_NAME, texts)
            embeddings = [d.embedding for d in response.data]

            collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            print(f"[Error] Failed to embed batch starting at index {i}: {e}")
        
# Get all chunks and call the embed_and_insert(chunk) function for all of them. With multiprocessing
def process_pdfs_and_insert(directory, batch_size=50):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            print(f"\n📄 Processing file: {filename}")
            chunks = chunk_pdf_recursive_token_size(pdf_path, parse_document=parse_document)

            batch_embed_and_insert(chunks, batch_size=batch_size)

            print(f"✅ Finished processing: {filename}")


# --------------------------------------------------------------------#
# --Parse, Tokenize, Chunk up, Embedd PDFs and insert into database---#
# --------------------------------------------------------------------#
process_pdfs_and_insert(PDF_DIRECTORY)
