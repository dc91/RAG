# check out the docs
# https://docs.trychroma.com/docs/overview/getting-started
# https://platform.openai.com/docs/guides/embeddings

# great videos about chroma and embedding with openai 
# https://www.youtube.com/watch?v=jbLa0KBW-jY


import fitz  # PyMuPDF
import tiktoken
import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

load_dotenv()

# -----------------------------------------------#
# -------------------Config----------------------#
# -----------------------------------------------# 
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PDF_DIRECTORY = "./okFile"
COLLECTION_NAME = "split_document_collection"
PERSIST_DIRECTORY = "split_document_storage"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
MAX_TOKENS = 512

# -----------------------------------------------#
# --------------Embedding and ChromaDB-----------#
# -----------------------------------------------#
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_KEY, model_name=EMBEDDING_MODEL_NAME
)

chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=openai_ef
)

client = OpenAI(api_key=OPENAI_KEY)


def chunk_pdf_by_tokens(pdf_path, model=EMBEDDING_MODEL_NAME, MAX_TOKENS=512):
    encoding = tiktoken.encoding_for_model(model)

    doc = fitz.open(pdf_path)
    chunks = []
    filename = os.path.basename(pdf_path)

    text_and_pagenumber = []  # List [(page_number, page_text)]
    for i, page in enumerate(doc):
        text = page.get_text()
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

    # Split into chunks of MAX_TOKENS
    total_chunks = (len(all_tokens) + MAX_TOKENS - 1) // MAX_TOKENS

    for i in range(total_chunks):
        start = i * MAX_TOKENS
        end = start + MAX_TOKENS
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

# -----------------------------------------------#
# -------------Process All PDFs------------------#
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

                # Get embedding
                print(f"Generating embedding for {chunk_id}")
                embedding = (
                    client.embeddings.create(
                        input=chunk_text, model=EMBEDDING_MODEL_NAME
                    )
                    .data[0]
                    .embedding
                )

                # Insert into ChromaDB
                print(f"Inserting chunk {chunk_id} into ChromaDB")
                collection.upsert(
                    ids=[chunk_id],
                    documents=[chunk_text],
                    embeddings=[embedding],
                    metadatas=[chunk["metadata"]],
                )

# Run the process on all PDFs
# process_pdfs_and_insert(PDF_DIRECTORY)

# -----------------------------------------------#
# -----------------Query Docs--------------------#
# -----------------------------------------------#
def query_documents(question, n_results=3):
    results = collection.query(query_texts=[question], n_results=n_results)

    # Extract the relevant chunks
    # Flatten the list of lists
    # results["documents"] is a list of lists, where each sublist corresponds to a document
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    for idx, document in enumerate(results["documents"][0]):
        doc_id = results["ids"][0][idx]
        distance = results["distances"][0][idx]
        metadata = results["metadatas"][0][idx]  # Include metadata if needed
        print("-" * 60)
        print(
            f"Found chunk: ID={doc_id}, Page={metadata.get('page_number')}, Distance={distance}"
        )
        print("-" * 60)
        print(f"Content:\n{document}\n\n---\n")

    return relevant_chunks

question = "vad säger vårdpersonal om att vara sjuksköterska?"
relevant_chunks = query_documents(question)



# --------------------------Not in this project scope--------------------------#
# ---------I basically straight up copied this part, but its fun to try--------#
# ----------------https://www.youtube.com/watch?v=vdLquGgg28A------------------#
# --------source: https://github.com/pdichone/rag-intro-chat-with-docs --------#


# -----------------------------------------------#
# -------------Response from OpenAI--------------#
# -----------------------------------------------#
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "Du är en assistent som svarar på frågor. Använd den information som finns i de "
        "angivna kontexten för att svara på din fråga. Om du inte kan svaret på frågan, "
        "säg att du inte vet svaret. Var kortfattad och koncis."
        "\nKontext:\n" + context + "\nFråga:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message
    return answer


# answer = generate_response(question, relevant_chunks)
# print(answer)
