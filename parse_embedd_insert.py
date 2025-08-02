# --This script takes all pdf files in PDF_DIRECTORY (default=pdf_data) and parses them.
# --It also tokenizes, Chunks up text, creates embeddings.
# --Chunking is specified in chunking.py
# --Then it inserts the embeddings to chromadb, with metadata.
# --Only nedds to be run if you want to add documents to the db.
# --Check config.py, before running, to make sure you have the right settings for your case.

import fitz  # PyMuPDF
import os
# import pymupdf4llm
from tqdm import tqdm

from config import (
    PDF_DIRECTORY,
    MD_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    USE_RECURSIVE_SPLIT,
    PARSE_AS_MD,
    NORMALIZE_AT_PARSE,
    get_collection,
    get_client
)
# from helping_scripts.chunking_kemi import chunk_pdf_recursive_token_size, chunk_pdf_by_tokens
from helping_scripts.chunking import chunk_pdf_recursive_token_size, chunk_pdf_by_tokens
from helping_scripts.norm_funcs import normalize_text

collection = get_collection() # set up db
client = get_client() # Client for embeddings


# -----------------------------------------------#
# --------------------Parse----------------------#
# -----------------------------------------------#
def parse_document(pdf_path, filename):
    doc = fitz.open(pdf_path)
    text_and_pagenumber = []  # List [(page_number, page_text)]

    for i, page in enumerate(doc):
        if PARSE_AS_MD:
            md_path = os.path.join(MD_DIRECTORY, filename[:-4] + f"_page{i+1}.md")
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # text = pymupdf4llm.to_markdown(doc, pages=[i])
        else:
            text = page.get_text(sort=True) # sort helps keep the right reading order in the page
        if text.strip():  # Skip empty pages
            if NORMALIZE_AT_PARSE:
                text = normalize_text(text)
            text_and_pagenumber.append((i + 1, text + " "))
    doc.close()
    return text_and_pagenumber


# -----------------------------------------------#
# -----Embedd PDFs and Insert to ChromaDB--------#
# -----------------------------------------------#
# Get embeddings of chunks from client, store with metadata in db

def embed_and_insert_batch(chunks, batch_size=50):
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding and inserting batches"):
        batch = chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]
        ids = [chunk["metadata"]["id"] for chunk in batch]

        try:
            response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL_NAME)
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
            print(f"\n📄 Parsing: {filename}")
            if USE_RECURSIVE_SPLIT:
                chunks = chunk_pdf_recursive_token_size(pdf_path, parse_document=parse_document)
            else:
                chunks = chunk_pdf_by_tokens(pdf_path, parse_document=parse_document)

            embed_and_insert_batch(chunks, batch_size=batch_size)

            print(f"✅ Finished processing: {filename}")
# --------------------------------------------------------------------#
# --Parse, Tokenize, Chunk up, Embedd PDFs and insert into database---#
# --------------------------------------------------------------------#
process_pdfs_and_insert(PDF_DIRECTORY)
