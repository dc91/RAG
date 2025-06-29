import fitz  # PyMuPDF
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import pathlib
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter

from config import (
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
MIN_CHUNK = 1024
MAX_CHUNK = 2048
CONTROL_SPACE_REGEX = re.compile(
    r'[\x00-\x1F\x7F\u00A0\u1680\u180E\u2000-\u200F\u2028\u2029\u202F\u205F\u2060\u2061\u2062\u2063\u2064\uFEFF]'
)

# Test to get 5 chunks from db and print them
# results = collection.get(limit=10, include=["documents", "metadatas"])
# for doc in results["documents"]:
#     print(doc)
# -----------------------------------------------#
# --------------------Parse----------------------#
# -----------------------------------------------#
def parse_to_md(pdf_directory = PDF_DIRECTORY):
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            
            filename_s = filename[:-4]  # Remove '.pdf'
            pdf_path = os.path.join(pdf_directory, filename)
            with fitz.open(pdf_path) as doc:
                num_pages = len(doc)
            for i in range(num_pages):
                
                file_path_md = os.path.join(md_base, f"{filename_s}_page_{i+1}.md")

                md_text = pymupdf4llm.to_markdown(
                    pdf_path,
                    write_images=False,
                    filename=f"{filename_s}",
                    pages=[i]
                )
                pathlib.Path(file_path_md).write_bytes(md_text.encode())
    
    # for filename in os.listdir(pdf_directory):
    #     if filename.endswith(".pdf"):
    #         filename_s = filename[:-4]  # Remove '.pdf'
    #         # pdf_path = os.path.join(pdf_directory, filename)
    #         file_path_md = os.path.join(md_base, filename[:-4] + ".md")

    #         md_text = pymupdf4llm.to_markdown(
    #             f"./{pdf_directory}/{filename}",
    #             write_images=False,
    #             filename=f"{filename_s}",
    #         )
    #         pathlib.Path(file_path_md).write_bytes(md_text.encode())
# -----------------------------------------------#
# -------------Tokenize and Chunk up-------------#
# -----------------------------------------------#
def clean_md_text(text):
    # Split text into lines
    text = re.sub(r"-[\u00AD\u200B\u200C\u200D\u200E\u200F]*\s*\n[\u00AD\u200B\u200C\u200D\u200E\u200F]*\s*", "", text)
    text = re.sub(r"[\u00AD\u200B\u200C\u200D\u200E\u200F]\s*", "", text)
    
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
    merged_text = " ".join(cleaned_lines)
    # Normalize whitespace
    merged_text = re.sub(r"\s+", " ", merged_text)
    # Clean up space before punctuation
    # merged_text = re.sub(r" +\.\s", ". ", merged_text)
    
    return CONTROL_SPACE_REGEX.sub('', merged_text).strip()

# def get_sentence_end(paragraph):
#     sentence_end = max(
#         (m.end() for m in re.finditer(r'(?<=[.!?])\s', paragraph[:MAX_CHUNK])),
#         default=None
#     )
#     if not sentence_end:
#         sentence_end = paragraph[:MAX_CHUNK].rfind('\n')
#     if sentence_end <= 0:
#         sentence_end = paragraph[:MAX_CHUNK].rfind(' ')
#     if sentence_end <= 0:
#         sentence_end = MAX_CHUNK
#     return sentence_end

# def split_large_paragraph(paragraph):
#     chunks = []
#     while len(paragraph) > MAX_CHUNK:
#         sentence_end = get_sentence_end(paragraph)
#         chunk = paragraph[:sentence_end].strip()
#         chunks.append(chunk)
#         paragraph = paragraph[sentence_end:].strip()
#     if paragraph:
#         chunks.append(paragraph)
#     return chunks

# def is_title_like(paragraph):
#     words = paragraph.strip().split()
#     return len(paragraph) < 50 and len(words) <= 6

# def is_page_number_like(paragraph):
#     words = paragraph.strip().split()
#     return len(paragraph) < 5 and len(words) <= 2

# def split_into_paragraphs(text):
#     lines = text.splitlines()
#     paragraphs = []
#     buffer = []
#     in_table = False

#     for line in lines:
#         stripped = line.strip()

#         if stripped.startswith("|"):
#             # Table row
#             buffer.append(line)
#             in_table = True
#         elif in_table and not stripped:
#             # Blank line ends the table
#             paragraphs.append("\n".join(buffer).strip())
#             buffer = []
#             in_table = False
#         elif in_table:
#             # Still in table
#             buffer.append(line)
#         elif not stripped:
#             # Blank line ends current paragraph
#             if buffer:
#                 paragraphs.append("\n".join(buffer).strip())
#                 buffer = []
#         else:
#             # Normal paragraph line
#             buffer.append(line)

#     # Add any trailing content
#     if buffer:
#         paragraphs.append("\n".join(buffer).strip())

#     return [p for p in paragraphs if p]

# def para_split(text):
#     full_text = []
#     growing_chunk = ""
#     title_buffer = ""
#     paragraphs = split_into_paragraphs(text)

#     for paragraph in paragraphs:
#         para_len = len(paragraph)

#         if paragraph.startswith("|"):
#             # Always flush before and after a table block
#             if growing_chunk.strip():
#                 full_text.append(growing_chunk.strip())
#                 growing_chunk = ""
#             full_text.append(paragraph)
#             continue

#         if is_title_like(paragraph):
#             title_buffer = paragraph
#             continue

#         if title_buffer:
#             paragraph = title_buffer + "\n\n" + paragraph
#             title_buffer = ""

#         if para_len < 50:
#             growing_chunk += paragraph + "\n\n"
#         elif para_len < MIN_CHUNK and len(growing_chunk) + para_len < MAX_CHUNK:
#             growing_chunk += paragraph + "\n\n"
#         else:
#             if growing_chunk.strip():
#                 full_text.append(growing_chunk.strip())
#                 growing_chunk = ""

#             if para_len > MAX_CHUNK:
#                 full_text.extend(split_large_paragraph(paragraph))
#             else:
#                 full_text.append(paragraph)

#     if title_buffer:
#         growing_chunk += title_buffer + "\n\n"

#     if growing_chunk.strip():
#         full_text.append(growing_chunk.strip())

#     return full_text

def remove_md_stuff(text):
    content = re.sub(r'\n#{1,6}|\n```\n.*?\n```|\n---+|\n___+', '', text)

    # Replace spaces with a single space
    content = re.sub(r' +', ' ', content)

    # Remove bold (handles **bold** and __bold__)
    content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', content)

    # Remove italic (handles *italic* and _italic_)
    content = re.sub(r'(\*|_)(.*?)\1', r'\2', content)
        
    return content


def chunk_pdf_by_paragraph(md_file):
    match = re.search(r'_page_(\d+)', md_file.name)
    page_number = int(match.group(1)) if match else None
    base_filename = re.sub(r'_page_\d+', '', md_file.name[:-3]) + ".pdf"
    splitter = MarkdownTextSplitter(chunk_size=MIN_CHUNK, chunk_overlap=0)

    # Read the .md file
    with md_file.open("r", encoding="utf-8") as f:
        markdown_text = f.read()

    # markdown_text = remove_md_stuff(markdown_text)
    # docs = para_split(markdown_text)
    # Split into chunks
    
    docs = splitter.split_text(markdown_text)
    
    chunks = []

    for i, chunk in enumerate(docs):
        chunk = remove_md_stuff(chunk)
        chunk = clean_md_text(chunk)
        chunk_metadata = {
            "id": f"{ md_file.name}_chunk{i + 1}",
            "filename":base_filename,
            "chunk_index": i + 1,
            "page_number": page_number,
            "total_chunks": len(docs)
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
        chunks = chunk_pdf_by_paragraph(md_file)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(embed_and_insert, chunk) for chunk in chunks]
            for future in as_completed(futures):
                future.result()

        print(f"✅ Finished processing: {md_file.name}")


# --This takes all pdf files in PDF_DIRECTORY and parses them.
# --It also tokenizes, Chunks up text, creates embeddings.
# --Then it inserts the embeddings to chromadb, with metadata.
# --Only nedds to be run if you want to add documents to db.
# --------------------------------------------------------------------#
# --Parse, Tokenize, Chunk up, Embedd PDFs and insert into database---#
# --------------------------------------------------------------------#
# parse_to_md(pdf_directory = PDF_DIRECTORY)
process_pdfs_and_insert()
