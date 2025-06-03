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
from tomlkit import (
    parse,
    dumps,
)  # allows to keep formatting, the toml package does not keep all formatting
# import toml # toml feels a bit faster though

load_dotenv()

# -----------------------------------------------#
# -------------------Config----------------------#
# -----------------------------------------------#
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PDF_DIRECTORY = "./okFile"
TOML_DIRECTORY = "questions/"
COLLECTION_NAME = "split_document_collection"
PERSIST_DIRECTORY = "split_document_storage"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
TOKEN_ENCODER = tiktoken.encoding_for_model(EMBEDDING_MODEL_NAME)
MAX_TOKENS = 512
# -----------------------------------------------#
# ------------ChromaDB and OpenAI Config---------#
# -----------------------------------------------#
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_KEY, model_name=EMBEDDING_MODEL_NAME
)

chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
# If you want to delete a collection
# chroma_client.delete_collection(name=COLLECTION_NAME)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=openai_ef
)

client = OpenAI(api_key=OPENAI_KEY)


# -----------------------------------------------#
# --------------------Parse----------------------#
# -----------------------------------------------#
def parse_document(pdf_path):
    doc = fitz.open(pdf_path)
    text_and_pagenumber = []  # List [(page_number, page_text)]

    for i, page in enumerate(doc):
        text = page.get_text(sort=True)
        if text.strip():  # Skip empty pages
            text_and_pagenumber.append((i + 1, text))
    doc.close()
    return text_and_pagenumber


# -----------------------------------------------#
# -------------Tokenize and Chunk up-------------#
# -----------------------------------------------#
def chunk_pdf_by_tokens(pdf_path, MAX_TOKENS=512):
    filename = os.path.basename(pdf_path)
    text_and_pagenumber = parse_document(pdf_path)  # List [(page_number, page_text)]
    chunks = []
    all_tokens = []
    token_page_map = []  # Keeps track of which page each token came from
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


# --------------------------------------------------------------#
# --------------------Embedd all questions----------------------#
# --------------------------------------------------------------#
def add_embeddings_to_toml(toml_dir):
    for filename in os.listdir(toml_dir):
        if filename.endswith(".toml"):
            with open(toml_dir + filename, "r", encoding="utf-8") as f:
                toml_f = parse(f.read())
            for question in toml_f["questions"]:
                question["question_embedding"] = (
                    client.embeddings.create(
                        input=question["question"], model=EMBEDDING_MODEL_NAME
                    )
                    .data[0]
                    .embedding
                )

            with open(toml_dir + "embedded_" + filename, "w", encoding="utf-8") as f:
                f.write(dumps(toml_f))


def get_embedded_questions(toml_dir):
    all_emdedded_questions = {}
    for filename in os.listdir(toml_dir):
        if filename.endswith(".toml"):
            with open(toml_dir + filename, "r", encoding="utf-8") as f:
                toml_f = parse(f.read())
            # print(toml_f["questions"][2]["files"])
            for question in toml_f["questions"]:
                q_id = question["id"]
                all_emdedded_questions[q_id] = question
    return all_emdedded_questions


# -----------------------------------------------#
# -----------------Query Docs--------------------#
# -----------------------------------------------#
def check_shrinking_matches(text_list, chunk, shrink_from_start=False):
        text_len = len(text_list)
        for i in range(text_len):
            current = text_list[i:] if shrink_from_start else text_list[:text_len - i]
            substring = "".join(current)
            if substring in chunk:
                idx = chunk.find(substring)
                percent_match = 100.0 * len(current) / text_len
                print(f"Found match {percent_match:.2f}%")
                print("Full text from answer: ", "".join(text_list))
                print("Text from answer: ", substring)
                print("Text from chunk: ", chunk[max(idx - 50, 0): min(len(chunk), idx + 200)])
                return True
        return False
    
def match_strings(chunk_text, answer):
    answer_chars = list(answer)
    print("Shrinking from end and matching")
    if check_shrinking_matches(answer_chars, chunk_text.replace("\n", " "), shrink_from_start=False):
        print("(Match from start)")
    else:
        print("(No match from start)")
    print("-" * 30)
    print("Shrinking from start and matching")
    # Then try shrinking from the left
    if check_shrinking_matches(answer_chars, chunk_text.replace("\n", " "), shrink_from_start=True):
        print("(Match from end)")
    else:
        print("(No match from end)")

def q_doc(question, n_results=3):
    results = collection.query(query_embeddings=[question["question_embedding"]], n_results=n_results)
    for idx, document in enumerate(results["documents"][0]):
        distance = results["distances"][0][idx]
        metadata = results["metadatas"][0][idx]  # Include metadata if needed
        print("-" * 60)
        print("-" * 60)
        print("File from result: ", metadata.get('filename'), "\nFile from toml", question["files"][0]["file"])
        print("Right file" if metadata.get("filename") == question["files"][0]["file"] else "Wrong file")
        print("Pages from result", question["files"][0]["page_numbers"], "\nPages from toml", metadata.get('page_number'))
        print("Right pages" if metadata.get("page_number") in question["files"][0]["page_numbers"] else "Wrong pages")
        print("Distance: ", distance)
        print("Question: ", question["question"])
        print("-" * 30)
        match_strings(document, question["answer"])
            
# Chroma will first embed each query text with the collection's embedding function, if query_texts is used
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


# --------------------------------------------------------------------#
# --Parse, Tokenize, Chunk up, Embedd PDFs and insert into database---#
# --------------------------------------------------------------------#
# process_pdfs_and_insert(PDF_DIRECTORY)

# --------------------------------------------------------------#
# ----------------Write a question, Run a query-----------------#
# --------------------------------------------------------------#
# question = "Hur introduceras asylbarnen till det svenska samhället på förskolan?"
# relevant_chunks = query_documents(question)

# --------------------------------------------------------------#
# -------Write new toml files with embeddings included----------#
# --------------------------------------------------------------#
# add_embeddings_to_toml(TOML_DIRECTORY)

# --------------------------------------------------------------#
# -------Get the data from toml files, with embedding-----------#
# --------------------------------------------------------------#
question_dict = get_embedded_questions(TOML_DIRECTORY)

# --------------------------------------------------------------#
# ------------------Run a query from toml files-----------------#
# --------------------------------------------------------------#
q_doc(question_dict["DC252"])


# --------------------------Not in this project scope--------------------------#
# ---------I basically straight up copied this part, but its fun to try--------#
# ----------------https://www.youtube.com/watch?v=vdLquGgg28A------------------#
# --------source: https://github.com/pdichone/rag-intro-chat-with-docs --------#


# -----------------------------------------------#
# -------------Response from OpenAI--------------#
# -----------------------------------------------#
# def generate_response(question, relevant_chunks):
#     context = "\n\n".join(relevant_chunks)
#     prompt = (
#         "Du är en assistent som svarar på frågor. Använd den information som finns i de "
#         "angivna kontexten för att svara på din fråga. Om du inte kan svaret på frågan, "
#         "säg att du inte vet svaret. Var kortfattad och koncis."
#         "\nKontext:\n" + context + "\nFråga:\n" + question
#     )

#     response = client.chat.completions.create(
#         model="gpt-4.1-nano",
#         messages=[
#             {
#                 "role": "system",
#                 "content": prompt,
#             },
#             {
#                 "role": "user",
#                 "content": question,
#             },
#         ],
#     )

#     answer = response.choices[0].message
#     return answer


# answer = generate_response(question, relevant_chunks)
# print(answer)
