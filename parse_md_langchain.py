import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import pathlib
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    MD_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    MIN_CHUNK_PARA,
    MAX_TOKENS,
    OVERLAP,
    get_collection,
    get_client,
)
from norm_funcs import clean_md_text, remove_md_stuff
from parasplit import merge_short_docs

collection = get_collection()
client = get_client()

# -----------------------------------------------#
# -------------Tokenize and Chunk up-------------#
# -----------------------------------------------#
def chunk_pdf_by_paragraph(md_file):
    match = re.search(r"_page(\d+)", md_file.name)
    page_number = int(match.group(1)) if match else None
    base_filename = re.sub(r"_page\d+", "", md_file.name[:-3]) + ".pdf"
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=EMBEDDING_MODEL_NAME,
        chunk_size=MAX_TOKENS,
        chunk_overlap=OVERLAP,
    )
    # Read the .md file
    with md_file.open("r", encoding="utf-8") as f:
        markdown_text = f.read()
    docs = splitter.split_text(markdown_text)
    if MIN_CHUNK_PARA > 0:
        docs = merge_short_docs(docs, min_length=MIN_CHUNK_PARA)

    chunks = []

    for i, chunk in enumerate(docs):
        chunk = remove_md_stuff(chunk)
        chunk = clean_md_text(chunk)
        chunk_index = i + 1
        chunk_metadata = {
            "id": f"{md_file.name}_chunk{i + 1}",
            "filename": base_filename,
            "chunk_index": chunk_index,
            "page_number": page_number,
            "total_chunks": len(docs),
        }
        chunks.append({"text": chunk, "metadata": chunk_metadata})
    return chunks


# -----------------------------------------------#
# -----Embedd PDFs and Insert to ChromaDB--------#
# -----------------------------------------------#
def embed_and_insert(chunk):
    chunk_id = chunk["metadata"]["id"]
    try:
        # print(f"[Thread] Embedding: {chunk_id}")
        embedding = (
            client.embeddings.create(input=chunk["text"], model=EMBEDDING_MODEL_NAME)
            .data[0]
            .embedding
        )

        # print(f"[Thread] Inserting: {chunk_id}")
        collection.upsert(
            ids=[chunk_id],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[chunk["metadata"]],
        )
    except Exception as e:
        print(f"[Error] Failed for {chunk_id}: {e}")


def get_max_workers(factor=1.5, fallback=4):
    try:
        return max(1, int(multiprocessing.cpu_count() * factor))
    except NotImplementedError:
        return fallback


def process_pdfs_and_insert(max_workers=None):
    md_files = list(pathlib.Path(MD_DIRECTORY).rglob("*.md"))
    if max_workers is None:
        max_workers = get_max_workers()

    for md_file in md_files:
        print(f"\n📄 Processing file: {md_file.name}")
        chunks = chunk_pdf_by_paragraph(md_file)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(embed_and_insert, chunk) for chunk in chunks]
            for future in as_completed(futures):
                future.result()

        print(f"✅ Finished processing: {md_file.name}")


# -------------------------------------------#
# --------------------Run--------------------#
# -------------------------------------------#
process_pdfs_and_insert()
