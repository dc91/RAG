# takes all pdf files in the folder and splits them into 512 token chunks.
# compares the regular text splitting with the the markdown conversion as well.
# When converting to markdown, the images are saved in a folder with the same name as the pdf file.

import fitz  # PyMuPDF
import tiktoken
import os
import pymupdf4llm
import pathlib

FOLDER_PATH = "okFile"
OUTPUT_BASE = "./compare_splits_sorted"

def chunk_pdf_by_tokens(pdf_path, model="text-embedding-3-small", max_tokens=512):
    encoding = tiktoken.encoding_for_model(model)

    doc = fitz.open(pdf_path)
    chunks = []
    filename = os.path.basename(pdf_path)

    text_and_pagenumber = []  # List [(page_number, page_text)]
    for i, page in enumerate(doc):
        text = page.get_text(sort=True)
        if text.strip():  # Skip empty pages
            text_and_pagenumber.append((i + 1, text))
    doc.close()

    # Combine text with page metadata and tokenize
    all_tokens = []
    token_page_map = []  # Keeps track of which page each token came from
    for page_number, page_text in text_and_pagenumber:
        tokens = encoding.encode(page_text)
        all_tokens.extend(tokens)
        token_page_map.extend([page_number] * len(tokens))

    # Split into chunks of max_tokens
    total_chunks = (len(all_tokens) + max_tokens - 1) // max_tokens

    for i in range(total_chunks):
        start = i * max_tokens
        end = start + max_tokens
        token_chunk = all_tokens[start:end]
        chunk_text = encoding.decode(token_chunk)

        # Majority page number for this chunk (for metadata)
        chunk_pages = token_page_map[start:end]
        if chunk_pages:
            most_common_page = max(set(chunk_pages), key=chunk_pages.count)
        else:
            most_common_page = None

        chunk_metadata = {
            "id": f"{filename}_chunk{i + 1}",
            "filename": filename,
            "page_number": most_common_page,
            "chunk_index": i + 1,
            "total_chunks": total_chunks,
        }

        chunks.append({"text": chunk_text, "metadata": chunk_metadata})

    return chunks


for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".pdf"):
        filename_s = filename[:-4]  # Remove '.pdf'
        pdf_path = os.path.join(FOLDER_PATH, filename)
        chunks = chunk_pdf_by_tokens(pdf_path)
        
        # Print the first two chunks for verification
        # for chunk in chunks[:2]:
        #     print("=" * 60)
        #     print("Metadata:", chunk["metadata"])
        #     print("=" * 60)
        #     print(chunk["text"])

        out_dir = os.path.join(OUTPUT_BASE, filename_s)
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
            f"./okFile/{filename}",
            write_images=True,
            filename=f"{filename_s}",
            image_path=f"./compare_splits/{filename_s}/",
        )
        pathlib.Path(file_path_md).write_bytes(md_text.encode())
