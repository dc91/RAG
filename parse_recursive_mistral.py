# --This script takes all pdf files in PDF_DIRECTORY (default=pdf_data) and parses them.
# --It also tokenizes, Chunks up text, creates embeddings.
# --Then it inserts the embeddings to chromadb, with metadata.
# --Only nedds to be run if you want to add documents to the db.
# --Check config.py, before running, to make sure you have the right settings for your case.

import fitz  # PyMuPDF
import os
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import multiprocessing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ratelimit import limits, sleep_and_retry

from config import (
    PDF_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    TOKEN_ENCODER,
    MAX_TOKENS,
    OVERLAP,
    USE_OPENAI,
    get_collection,
    get_client
)


collection = get_collection() # set up db
client = get_client() # OpenAI client for embeddings


# -----------------------------------------------#
# --------------------Parse----------------------#
# -----------------------------------------------#
def parse_document(pdf_path):
    doc = fitz.open(pdf_path)
    text_and_pagenumber = []  # List [(page_number, page_text)]

    for i, page in enumerate(doc):
        text = page.get_text(sort=True) # sort helps keep the right reading order in the page
        if text.strip():  # Skip empty pages
            text_and_pagenumber.append((i + 1, text + " "))
    doc.close()
    return text_and_pagenumber



# -----------------------------------------------#
# -------------Tokenize and Chunk up-------------#
# -----------------------------------------------#
def get_token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = TOKEN_ENCODER
    if USE_OPENAI:
        num_tokens = len(encoding.encode(string, disallowed_special=()))
    else:
        num_tokens = len(encoding.encode(text=string, add_special_tokens=False))
    return num_tokens

def chunk_pdf_by_paragraph_tokens(pdf_path, MAX_TOKENS=MAX_TOKENS, OVERLAP=OVERLAP):
    filename = os.path.basename(pdf_path)
    text_and_pagenumber = parse_document(pdf_path)  # [(page_number, page_text)]
    
    splitter = RecursiveCharacterTextSplitter(
        length_function=get_token_count,
        chunk_size=MAX_TOKENS,
        chunk_overlap=OVERLAP,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )

    all_paragraphs = []
    paragraph_page_map = []

    for page_number, page_text in text_and_pagenumber:
        # Split by paragraphs on this page
        paragraph_chunks = splitter.split_text(page_text)
        all_paragraphs.extend(paragraph_chunks)
        paragraph_page_map.extend([page_number] * len(paragraph_chunks))

    # Now merge paragraphs into token-bounded chunks
    chunks = []
    current_chunk = []
    current_token_count = 0
    chunk_index = 1

    def finalize_chunk():
        nonlocal current_chunk, current_token_count, chunk_index
        if not current_chunk:
            return
        chunk_text = " ".join(current_chunk)
        token_chunk = TOKEN_ENCODER.encode(chunk_text, add_special_tokens=False)
        page_list = sorted(set(chunk_page_numbers))
        chunk_metadata = {
            "id": f"{filename}_chunk{chunk_index}",
            "filename": filename,
            "page_number": ",".join(map(str, page_list)),
            "chunk_index": chunk_index,
        }
        chunks.append({
            "text": TOKEN_ENCODER.decode(token_chunk),
            "metadata": chunk_metadata,
        })
        chunk_index += 1
        current_chunk = []
        current_token_count = 0

    chunk_page_numbers = []

    for paragraph, page_number in zip(all_paragraphs, paragraph_page_map):
        tokens = TOKEN_ENCODER.encode(paragraph, add_special_tokens=False)
        if current_token_count + len(tokens) > MAX_TOKENS:
            finalize_chunk()
            chunk_page_numbers = []
        current_chunk.append(paragraph)
        current_token_count += len(tokens)
        chunk_page_numbers.append(page_number)

    finalize_chunk()  # Catch the last one

    total_chunks = len(chunks)
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = total_chunks

    return chunks


# -----------------------------------------------#
# -----Embedd PDFs and Insert to ChromaDB--------#
# -----------------------------------------------#
MAX_CALLS_PER_SECOND = 6

@sleep_and_retry
@limits(calls=MAX_CALLS_PER_SECOND, period=1)
def call_embedding_api(model, inputs):
    return client.embeddings.create(model=model, inputs=inputs)

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
            chunks = chunk_pdf_by_paragraph_tokens(pdf_path)

            batch_embed_and_insert(chunks, batch_size=batch_size)

            print(f"✅ Finished processing: {filename}")


# --------------------------------------------------------------------#
# --Parse, Tokenize, Chunk up, Embedd PDFs and insert into database---#
# --------------------------------------------------------------------#
process_pdfs_and_insert(PDF_DIRECTORY)
