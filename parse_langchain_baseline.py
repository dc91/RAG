# --This script takes all pdf files in PDF_DIRECTORY (default=pdf_data) and parses them.
# --It also tokenizes, Chunks up text, creates embeddings.
# --Then it inserts the embeddings to chromadb, with metadata.
# --Only nedds to be run if you want to add documents to the db.
# --Check config.py, before running, to make sure you have the right settings for your case.

import fitz  # PyMuPDF
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from langchain.text_splitter import RecursiveCharacterTextSplitter


from config import (
    PDF_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    TOKEN_ENCODER,
    MAX_TOKENS,
    OVERLAP,
    get_collection,
    get_client
)
# from norm_funcs import normalize_text

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
            # norm_text = normalize_text(text)
            text_and_pagenumber.append((i + 1, text + " "))
    doc.close()
    return text_and_pagenumber



# -----------------------------------------------#
# -------------Tokenize and Chunk up-------------#
# -----------------------------------------------#
def chunk_pdf_by_paragraph_tokens(pdf_path, MAX_TOKENS=MAX_TOKENS, OVERLAP=OVERLAP):
    filename = os.path.basename(pdf_path)
    text_and_pagenumber = parse_document(pdf_path)  # [(page_number, page_text)]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=EMBEDDING_MODEL_NAME,
        chunk_size=MAX_TOKENS,
        chunk_overlap=OVERLAP,
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
        token_chunk = TOKEN_ENCODER.encode(chunk_text)
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
        tokens = TOKEN_ENCODER.encode(paragraph)
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
def get_max_workers(factor=1.5, fallback=4):
    try:
        return max(1, int(multiprocessing.cpu_count() * factor))
    except NotImplementedError:
        return fallback

# Get embeddings of chunks from client, store with metadata in db
def embed_and_insert(chunk):
    chunk_id = chunk["metadata"]["id"]
    try:
        embedding = client.embeddings.create(
            input=chunk["text"], model=EMBEDDING_MODEL_NAME
        ).data[0].embedding

        collection.upsert(
            ids=[chunk_id],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[chunk["metadata"]],
        )
    except Exception as e:
        print(f"[Error] Failed for {chunk_id}: {e}")
        
# Get all chunks and call the embed_and_insert(chunk) function for all of them. With multiprocessing
def process_pdfs_and_insert(directory, max_workers=None):
    if max_workers is None:
        max_workers = get_max_workers()

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            print(f"\n📄 Processing file: {filename}")
            chunks = chunk_pdf_by_paragraph_tokens(pdf_path)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(embed_and_insert, chunk) for chunk in chunks]
                for future in as_completed(futures):
                    future.result()

            print(f"✅ Finished processing: {filename}")


# --------------------------------------------------------------------#
# --Parse, Tokenize, Chunk up, Embedd PDFs and insert into database---#
# --------------------------------------------------------------------#
process_pdfs_and_insert(PDF_DIRECTORY)
