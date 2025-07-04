import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from config import (
    PDF_DIRECTORY,
    MD_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    TOKEN_ENCODER,
    MAX_TOKENS,
    OVERLAP,
    get_collection,
    get_client
)
from norm_funcs import clean_md_text, remove_md_stuff

collection = get_collection() # set up db
client = get_client() # OpenAI client for embeddings


# -----------------------------------------------#
# --------------------Parse----------------------#
# -----------------------------------------------#

def parse_document_from_md(MD_DIRECTORY, filename_base):
    text_and_pagenumber = []
    
    # Regex to match page numbers in filenames like 'report_2023_page1.md'
    pattern = re.compile(rf"^{re.escape(filename_base)}_page(\d+)\.md$")

    for fname in os.listdir(MD_DIRECTORY):
        match = pattern.match(fname)
        if match:
            page_num = int(match.group(1))
            md_path = os.path.join(MD_DIRECTORY, fname)
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if text.strip():
                    norm_text = remove_md_stuff(text)
                    norm_text = clean_md_text(norm_text)
                    text_and_pagenumber.append((page_num, norm_text + " "))
    
    # Sort by page number in case files were read out of order.
    text_and_pagenumber.sort(key=lambda x: x[0])
    
    return text_and_pagenumber
# -----------------------------------------------#
# -------------Tokenize and Chunk up-------------#
# -----------------------------------------------#
def chunk_pdf_by_tokens(pdf_path, MAX_TOKENS=MAX_TOKENS, OVERLAP=OVERLAP):
    filename = os.path.basename(pdf_path)
    filename_base = os.path.splitext(filename)[0]  # Remove .pdf extension

    # Load from Markdown files, not the PDF
    text_and_pagenumber = parse_document_from_md(MD_DIRECTORY, filename_base)

    chunks = []
    all_tokens = []
    token_page_map = []  # Keeps track of which page each token came from

    for page_number, page_text in text_and_pagenumber:
        tokens = TOKEN_ENCODER.encode(page_text)
        all_tokens.extend(tokens)
        token_page_map.extend([page_number] * len(tokens))

    step = MAX_TOKENS - OVERLAP
    total_tokens = len(all_tokens)
    i = 0
    chunk_index = 1

    while i < total_tokens:
        start = i
        end = min(i + MAX_TOKENS, total_tokens)
        token_chunk = all_tokens[start:end]
        chunk_text = TOKEN_ENCODER.decode(token_chunk)
        chunk_pages = token_page_map[start:end]
        page_list = sorted(set(chunk_pages))
        chunk_metadata = {
            "id": f"{filename}_chunk{chunk_index}",
            "filename": filename,
            "page_number": ",".join(map(str, page_list)),
            "chunk_index": chunk_index,
        }

        chunks.append({"text": chunk_text, "metadata": chunk_metadata})
        i += step
        chunk_index += 1

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
            print(f"\n📄 Processing md version of file: {filename}")
            chunks = chunk_pdf_by_tokens(pdf_path)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(embed_and_insert, chunk) for chunk in chunks]
                for future in as_completed(futures):
                    future.result()

            print(f"✅ Finished processing: {filename}")


# --------------------------------------------------------------------#
# --Parse, Tokenize, Chunk up, Embedd PDFs and insert into database---#
# --------------------------------------------------------------------#
process_pdfs_and_insert(PDF_DIRECTORY)
