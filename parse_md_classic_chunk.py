# --This script takes all pdf files in PDF_DIRECTORY (default=pdf_data) and parses them.
# --It also tokenizes, Chunks up text, creates embeddings.
# --Then it inserts the embeddings to chromadb, with metadata.
# --Only nedds to be run if you want to add documents to the db.
# --Check config.py, before running, to make sure you have the right settings for your case.

# import pathlib
import fitz  # PyMuPDF
import pymupdf4llm
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from config import (
    PDF_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    TOKEN_ENCODER,
    MAX_TOKENS,
    OVERLAP,
    get_collection,
    get_client
)
MD_DIRECTORY = "md_data"
md_base = MD_DIRECTORY + "/md"
collection = get_collection() # set up db
client = get_client() # OpenAI client for embeddings


# -----------------------------------------------#
# --------------------Parse----------------------#
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

def normalize_text(input_text):
    # Remove split words at the end of lines
    normalized = re.sub(r"- ?\n", "", input_text.strip())
    # Replace any sequence of whitespace (including newlines) with a single space
    normalized = re.sub(r"\s+", " ", normalized)
    # Don't keep space if end of sentence
    normalized = re.sub(r" +\.\s", ". ", normalized)

    return normalized

def parse_document(pdf_path):
    doc = fitz.open(pdf_path)
    text_and_pagenumber = []  # List [(page_number, page_text)]

    for i, page in enumerate(doc):
        text = pymupdf4llm.to_markdown(
                    pdf_path,
                    write_images=False,
                    pages=[i]
                )
        # text = page.get_text(sort=True) # sort helps keep the right reading order in the page
        if text.strip():  # Skip empty pages
            norm_text = normalize_text(text)
            norm_text = remove_md_stuff(text)
            norm_text = clean_md_text(text)
            text_and_pagenumber.append((i + 1, norm_text + " "))
    doc.close()
    return text_and_pagenumber

def parse_document_from_md(md_base, filename_base):
    text_and_pagenumber = []
    
    # Regex to match page numbers in filenames like 'report_2023_page_1.md'
    pattern = re.compile(rf"^{re.escape(filename_base)}_page(\d+)\.md$")

    for fname in os.listdir(md_base):
        match = pattern.match(fname)
        if match:
            page_num = int(match.group(1))
            md_path = os.path.join(md_base, fname)
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if text.strip():
                    norm_text = normalize_text(text)
                    norm_text = remove_md_stuff(norm_text)
                    norm_text = clean_md_text(norm_text)
                    text_and_pagenumber.append((page_num, norm_text + " "))
    
    # Sort by page number in case files were read out of order
    text_and_pagenumber.sort(key=lambda x: x[0])
    
    return text_and_pagenumber
# -----------------------------------------------#
# -------------Tokenize and Chunk up-------------#
# -----------------------------------------------#
def chunk_pdf_by_tokens(pdf_path, MAX_TOKENS=MAX_TOKENS, OVERLAP=OVERLAP):
    filename = os.path.basename(pdf_path)
    filename_base = os.path.splitext(filename)[0]  # Remove .pdf extension

    # Load from Markdown files, not the PDF
    text_and_pagenumber = parse_document_from_md(md_base, filename_base)

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
            print(f"\n📄 Processing file: {filename}")
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
