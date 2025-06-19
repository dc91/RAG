import fitz  # PyMuPDF
import tiktoken
import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
import re
from config import (
    OPENAI_KEY,
    PDF_DIRECTORY,
    COLLECTION_NAME,
    PERSIST_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    TOKEN_ENCODER,
    MAX_TOKENS,
)

load_dotenv()

# -----------------------------------------------#
# ------------ChromaDB and OpenAI Config---------#
# -----------------------------------------------#
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_KEY, model_name=EMBEDDING_MODEL_NAME
)
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=openai_ef
)
client = OpenAI(api_key=OPENAI_KEY)


# -----------------------------------------------#
# --------------------Parse----------------------#
# -----------------------------------------------#
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
        text = page.get_text(sort=True)
        if text.strip():  # Skip empty pages
            norm_text = normalize_text(text)
            text_and_pagenumber.append((i + 1, norm_text + " "))
    doc.close()
    # Test print
    # print(text_and_pagenumber)
    return text_and_pagenumber
# Test call
# parse_document(pdf_path = os.path.join(PDF_DIRECTORY, "12_BERÄTTELSER_OM_SKAM.pdf"))


# -----------------------------------------------#
# -------------Tokenize and Chunk up-------------#
# -----------------------------------------------#
def chunk_pdf_by_tokens(pdf_path, MAX_TOKENS=512):
    filename = os.path.basename(pdf_path)
    text_and_pagenumber = parse_document(pdf_path)  # List [(page_number, page_text)]
    chunks = []
    all_tokens = []
    token_page_map = []  # Keeps track of which page each token came from, [page number of token1, token2, token3 ...]
    for page_number, page_text in text_and_pagenumber:
        tokens = TOKEN_ENCODER.encode(page_text)
        all_tokens.extend(tokens)
        token_page_map.extend([page_number] * len(tokens))

    # Split into chunks of MAX_TOKENS
    total_chunks = (len(all_tokens) + MAX_TOKENS - 1) // MAX_TOKENS

    for i in range(total_chunks):
        start = i * MAX_TOKENS
        end = start + MAX_TOKENS
        token_chunk = all_tokens[start:end]
        chunk_text = TOKEN_ENCODER.decode(token_chunk)

        # Majority page number for this chunk (for metadata)
        chunk_pages = token_page_map[start:end]
        page_list = sorted(set(chunk_pages))
        # if chunk_pages:
        #     most_common_page = max(set(chunk_pages), key=chunk_pages.count)
        # else:
        #     most_common_page = None

        chunk_metadata = {
            "id": f"{filename}_chunk{i + 1}",
            "filename": filename,
            "page_number": ",".join(map(str, page_list)),
            "chunk_index": i + 1,
            "total_chunks": total_chunks,
        }

        chunks.append({"text": chunk_text, "metadata": chunk_metadata})
        # Test print
        # print("Chunk", i, ": ", chunk_text)
    return chunks
# Test call
# chunk_pdf_by_tokens(os.path.join(PDF_DIRECTORY, "12_BERÄTTELSER_OM_SKAM.pdf"))


# -----------------------------------------------#
# -----Embedd PDFs and Insert to ChromaDB--------#
# -----------------------------------------------#
def process_pdfs_and_insert(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            print(f"Processing: {filename}")
            pdf_path = os.path.join(directory, filename)
            chunks = chunk_pdf_by_tokens(pdf_path)

            for chunk in chunks:
                chunk_id = chunk["metadata"]["id"]
                chunk_text = chunk["text"]
                # Test print
                # print("Chunk", chunk_id, ": ", chunk_text)
                # Get embedding
                print(f"Generating embedding for {chunk_id}")
                embedding = (
                    client.embeddings.create(
                        input=chunk_text, model=EMBEDDING_MODEL_NAME
                    )
                    .data[0]
                    .embedding
                )
                # Insert into ChromaDB, upsert to not upload existing files
                print(f"Inserting chunk {chunk_id} into ChromaDB")
                collection.upsert(
                    ids=[chunk_id],
                    documents=[chunk_text],
                    embeddings=[embedding],
                    metadatas=[chunk["metadata"]],
                )


# --This takes all pdf files in PDF_DIRECTORY and parses them.
# --It also tokenizes, Chunks up text, creates embeddings.
# --Then it inserts the embeddings to chromadb, with metadata.
# --Only nedds to be run if you want to add documents to db.
# --------------------------------------------------------------------#
# --Parse, Tokenize, Chunk up, Embedd PDFs and insert into database---#
# --------------------------------------------------------------------#
process_pdfs_and_insert(PDF_DIRECTORY)
