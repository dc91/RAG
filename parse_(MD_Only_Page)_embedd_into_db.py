import fitz  # PyMuPDF
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import pathlib
import pymupdf4llm
# from langchain.text_splitter import MarkdownTextSplitter

from config import (
    # MAX_TOKENS,
    PDF_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    get_collection,
    get_client
)

collection = get_collection()
client = get_client()

MD_DIRECTORY = "md_data"
md_base = MD_DIRECTORY + "/md"
os.makedirs(md_base, exist_ok=True)

# -----------------------------------------------#
# --------------------Parse----------------------#
# -----------------------------------------------#
def parse_to_md(pdf_directory = PDF_DIRECTORY):    
    for filename in os.listdir(PDF_DIRECTORY):
        if filename.endswith(".pdf"):
            
            filename_s = filename[:-4]  # Remove '.pdf'
            pdf_path = os.path.join(PDF_DIRECTORY, filename)
            with fitz.open(pdf_path) as doc:
                num_pages = len(doc)
            for i in range(num_pages):
                
                file_path_md = os.path.join(md_base, f"{filename_s}_page_{i+1}.md")

                md_text = pymupdf4llm.to_markdown(
                    f"./{PDF_DIRECTORY}/{filename}",
                    write_images=False,
                    filename=f"{filename_s}",
                    pages=[i]
                )
                pathlib.Path(file_path_md).write_bytes(md_text.encode())
# -----------------------------------------------#
# -------------Tokenize and Chunk up-------------#
# -----------------------------------------------#
def clean_md_text(text):
    CONTROL_SPACE_REGEX = re.compile(
        r'[\x00-\x1F\x7F\u00A0\u1680\u180E\u2000-\u200F\u2028\u2029\u202F\u205F\u2060\u2061\u2062\u2063\u2064\uFEFF]'
    )
    text = re.sub(r"-[\u00AD\u200B\u200C\u200D\u200E\u200F]*\s*\n[\u00AD\u200B\u200C\u200D\u200E\u200F]*\s*", "", text)
    text = re.sub(r"[\u00AD\u200B\u200C\u200D\u200E\u200F]\s*", "", text)
    # Split text into lines
    lines = text.split('\n')
    
    cleaned_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        # Skip lines that only contain a number (e.g., page numbers)
        if re.fullmatch(r'\s*\d+\s*', line):
            continue
        # Skip empty lines
        if not stripped_line:
            continue
        cleaned_lines.append(line)
    
    # Join all lines into one paragraph-like text
    merged_text = '\n'.join(cleaned_lines)
    merged_text = re.sub(r"\s+", " ", merged_text)
    
    return CONTROL_SPACE_REGEX.sub('', merged_text).strip()

def remove_md_stuff(text):
    content = re.sub(r'\n#{1,6}|\n```\n.*?\n```|\n---+|\n___+', '', text)

    # Replace spaces with a single space
    content = re.sub(r' +', ' ', content)

    # Remove bold (handles **bold** and __bold__)
    content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', content)

    # Remove italic (handles *italic* and _italic_)
    content = re.sub(r'(\*|_)(.*?)\1', r'\2', content)
        
    return content


def chunk_pdf_by_page(md_file):
    match = re.search(r'_page_(\d+)', md_file.name)
    page_number = int(match.group(1)) if match else None
    base_filename = md_file.name[:-3] + ".pdf"
    # splitter = MarkdownTextSplitter(chunk_size=MAX_TOKENS, chunk_overlap=0)

    # Read the .md file
    with md_file.open("r", encoding="utf-8") as f:
        markdown_text = f.read()

    markdown_text = remove_md_stuff(markdown_text)
    # docs = para_split(markdown_text)
    # Split into chunks
    
    # docs = splitter.split_text(remove_md_stuff(markdown_text))
    chunks = []
    chunk_metadata = {
        "id": f"{ md_file.name}_chunk{1}",
        "filename":base_filename,
        "chunk_index": 1,
        "page_number": page_number,
        "total_chunks": 1
    }
    chunks.append({"text": clean_md_text(markdown_text), "metadata": chunk_metadata})
    
    return chunks


# -----------------------------------------------#
# -----Embedd PDFs and Insert to ChromaDB--------#
# -----------------------------------------------#
def embed_and_insert(chunk):
    chunk_id = chunk["metadata"]["id"]
    try:
        # print(f"[Thread] Embedding: {chunk_id}")
        embedding = client.embeddings.create(
            input=chunk["text"], model=EMBEDDING_MODEL_NAME
        ).data[0].embedding

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
    md_files = list(pathlib.Path(md_base).rglob("*.md"))
    if max_workers is None:
        max_workers = get_max_workers()

    for md_file in md_files:
        print(f"\n📄 Processing file: {md_file.name}")
        chunks = chunk_pdf_by_page(md_file)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(embed_and_insert, chunk) for chunk in chunks]
            for future in as_completed(futures):
                future.result()

        # print(f"✅ Finished processing: {md_file.name}")


# --This takes all pdf files in PDF_DIRECTORY and parses them.
# --It also tokenizes, Chunks up text, creates embeddings.
# --Then it inserts the embeddings to chromadb, with metadata.
# --Only nedds to be run if you want to add documents to db.
# --------------------------------------------------------------------#
# --Parse, Tokenize, Chunk up, Embedd PDFs and insert into database---#
# --------------------------------------------------------------------#
# parse_to_md(pdf_directory = PDF_DIRECTORY)
process_pdfs_and_insert()
