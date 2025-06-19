# takes all pdf files in the folder and splits them into 512 token chunks.
# compares the regular text splitting with the the markdown conversion as well.
# When converting to markdown, the images are saved in a folder with the same name as the pdf file.

import fitz  # PyMuPDF
import os
import pymupdf4llm
import pathlib
import re

from config import (
    PDF_DIRECTORY,
    TOKEN_ENCODER,
    MAX_TOKENS,
    OVERLAP,
    OUTPUT_DIRECTORY_COMPARE_SPLITS
)

def normalize_text(input_text):
    # Remove split words at the end of lines
    normalized = re.sub(r"- ?\n", "", input_text.strip())
    # Replace any sequence of whitespace (including newlines) with a single space
    normalized = re.sub(r"\s+", " ", normalized)
    # Don't keep space if end of sentence
    normalized = re.sub(r" +\.\s", ". ", normalized) 
    
    return normalized

def chunk_pdf_by_tokens(pdf_path, model="text-embedding-3-small", MAX_TOKENS=MAX_TOKENS, OVERLAP=OVERLAP):
    # encoding = tiktoken.encoding_for_model(model)

    doc = fitz.open(pdf_path)
    chunks = []
    filename = os.path.basename(pdf_path)

    text_and_pagenumber = []  # List [(page_number, page_text)]
    for i, page in enumerate(doc):
        text = page.get_text(sort=True)
        if text.strip():  # Skip empty pages
            norm_text = normalize_text(text)
            text_and_pagenumber.append((i + 1, norm_text + " "))
    doc.close()

    # Combine text with page metadata and tokenize
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
    
    #
    return chunks


for filename in os.listdir(PDF_DIRECTORY):
    if filename.endswith(".pdf"):
        filename_s = filename[:-4]  # Remove '.pdf'
        pdf_path = os.path.join(PDF_DIRECTORY, filename)
        chunks = chunk_pdf_by_tokens(pdf_path)
        
        # Print the first two chunks for verification
        # for chunk in chunks[:2]:
        #     print("=" * 60)
        #     print("Metadata:", chunk["metadata"])
        #     print("=" * 60)
        #     print(chunk["text"])

        out_dir = os.path.join(OUTPUT_DIRECTORY_COMPARE_SPLITS, filename_s)
        os.makedirs(out_dir, exist_ok=True)
        file_path_txt = os.path.join(out_dir, "splitToText.txt")
        file_path_md = os.path.join(out_dir, "splitToMd.md")

        with open(file_path_txt, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write("=" * 60 + "\n")
                f.write(f"Metadata: {chunk['metadata']}\n")
                f.write("=" * 60 + "\n")
                f.write(chunk["text"] + "\n\n")

        md_text = pymupdf4llm.to_markdown(
            f"./{PDF_DIRECTORY}/{filename}",
            write_images=True,
            filename=f"{filename_s}",
            image_path=f"{OUTPUT_DIRECTORY_COMPARE_SPLITS}/{filename_s}/",
        )
        pathlib.Path(file_path_md).write_bytes(md_text.encode())
