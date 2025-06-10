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
)  # allows to keep formatting, but slow. So only used for creating toml files
import tomli # Used to read the toml files fast.
import pandas as pd
import re

load_dotenv()

# -----------------------------------------------#
# -------------------Config----------------------#
# -----------------------------------------------#
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PDF_DIRECTORY = "./pdf_data"
TOML_DIRECTORY = "questions/embedded"
COLLECTION_NAME = "document_collection_norm_all"
PERSIST_DIRECTORY = "document_storage_norm_all"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
TOKEN_ENCODER = tiktoken.encoding_for_model(EMBEDDING_MODEL_NAME)
MAX_TOKENS = 512
RESULTS_PER_QUERY = 3
MATCH_THRESHOLD = 50
RESULTS_CSV_NAME = "results/norm_queries.csv"
RESULTS_EXCEL_NAME = "results/norm_queries_excel.xlsx"
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

# Test to get 5 chunks from db and print them
# results = collection.get(limit=5)
# for doc in results["documents"]:
#     print(doc)
    
    
# ------------------------------------------------------------------------#
# --------------------Function definitions--------------------------------#
# ---------------(Calls are done after all definitions)-------------------#

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
# Test call
# process_pdfs_and_insert(PDF_DIRECTORY)


# --------------------------------------------------------------#
# --------------------Embedd all questions----------------------#
# --------------------------------------------------------------#
# Creates new toml files with the embedding of the question added
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

# Reads the toml files, with the embedded questions
def get_embedded_questions(toml_dir):
    all_embedded_questions = {}
    for filename in os.listdir(toml_dir):
        if filename.endswith(".toml") and "embedded_" in filename:
            file_path = os.path.join(toml_dir, filename)
            with open(file_path, "rb") as f:  # tomli requires binary mode
                toml_data = tomli.load(f)
            questions = toml_data.get("questions", [])
            for question in questions:
                q_id = question.get("id")
                if q_id:
                    all_embedded_questions[q_id] = question
    return all_embedded_questions


# -----------------------------------------------#
# -----------------Query Docs--------------------#
# -----------------------------------------------#

# --------------Helping Functions----------------#
# -----------------------------------------------#

def check_shrinking_matches(text_list, chunk, shrink_from_start=False, text_or_embedding="embedding"):
        text_len = len(text_list)
        for i in range(text_len - 1):
            current = text_list[i:] if shrink_from_start else text_list[:text_len - i]
            substring = "".join(current)
            if substring in chunk.lower():
                percent_match = 100.0 * len(current) / text_len
                if text_or_embedding == "text":
                    idx = chunk.find(substring)
                    print(f"Found match {percent_match:.2f}%")
                    print(f"Match starts at char position: {idx}")
                    print(f"Match ends at char position: {idx + len(substring)}")
                    print(f"Match length: {len(substring)}")
                    print("\nMatch from answer: ", substring)
                    print("\nPiece from chunk: ", chunk)
                return True, percent_match, len(substring)
        return False, 0, 0
    
    
    
# This is just a function that calls check_shrinking_matches, and prints stuff around it
# Only used with query_documents_one_embedding
def match_strings(chunk_text, answer):
    answer_chars = list(answer.lower())
    print("[Shrinking from end and matching...]")
    match_from_start_bool = check_shrinking_matches(answer_chars, chunk_text, shrink_from_start=False, text_or_embedding="text")[0]
    if match_from_start_bool:
        print("(Match from start)")
    else:
        print("(No match from start)")
    print("-" * 30)
    print("[Shrinking from start and matching...]")
    match_from_end_bool = check_shrinking_matches(answer_chars, chunk_text, shrink_from_start=True, text_or_embedding="text")[0]
    if match_from_end_bool:
        print("(Match from end)")
    else:
        print("(No match from end)")
    return match_from_start_bool, match_from_end_bool

# Only concerns the excel file output
def escape_excel_formulas(val):
    if isinstance(val, str) and val.startswith("="):
        return "'" + val # Maybe find another way? Because this kind of changes the chunk. It adds ' to chunks with = in the beginning.
    return val

def save_data_from_result(all_rows, all_columns, csv_name, excel_name):
    df = pd.DataFrame(all_rows, columns=all_columns)
    df.to_csv(csv_name, encoding="utf-8", index=False)
    
    df = df.map(escape_excel_formulas)
    # This is needed since some chunks start with '=' which excel interprets as a formula.
    # The loop below can check for instances of chunks starting with '='
    # for i, row in enumerate(all_rows):
    #     for j, val in enumerate(row):
    #         if isinstance(val, str) and val.strip().startswith("="):
    #             print(f"Warning: Cell at row {i}, column {j} starts with '=' and may be interpreted as a formula")
    with pd.ExcelWriter(excel_name) as writer:  
        df.to_excel(writer, sheet_name="Test_Query", index=False)


# ----------------Query Functions----------------#
# -----------------------------------------------#
# There are 3 different ones. 
# (query_documents_all_embeddings) is for query with all embeddings,
# (query_documents_one_embedding) is for a single embedding query,
# (query_documents_text_input) is for a single text query.

def query_documents_all_embeddings(question, n_results=3):
    all_columns = ["Result_Id", "Correct_File", "Guessed_File", "Filename_Match", "Correct_Pages", "Guessed_Page", "Page_Match", "Distance", "Text_Match_Start_Percent", "Match_Length_Start", "Text_Match_End_Percent", "Match_Length_End", "No_match", "Match_Threshold", "Difficulty", "Category", "Expected_answer", "Question", "Returned_Chunk", "Chunk_Id"]
    all_rows = []
    for value in question.values():
        results = collection.query(query_embeddings=[value["question_embedding"]], n_results=n_results)
        for idx, document in enumerate(results["documents"][0]): # document here refers to chunks, due to chromadb naming
            distance = results["distances"][0][idx]
            metadata = results["metadatas"][0][idx]
            
            correct_file = value["files"][0]["file"]
            guessed_file = metadata.get("filename")
            filename_match = guessed_file == correct_file

            correct_pages = value["files"][0]["page_numbers"]
            guessed_page = metadata.get("page_number")
            # Don't check for page matches if wrong file
            if filename_match:
                page_match = guessed_page in correct_pages
            else:
                page_match = False
            
            match_from_start_bool, match_from_start_float, match_from_start_length = check_shrinking_matches(list(value["answer"].lower()), document, shrink_from_start=False)
            match_from_end_bool, match_from_end_float, match_from_end_length = check_shrinking_matches(list(value["answer"].lower()), document, shrink_from_start=True)
            # We need to figure out what the thershold is, and how to calculate it. This adds both matches.
            # We could use match length somehow as well?
            match_threshold = True if (match_from_start_float + match_from_end_float > MATCH_THRESHOLD) else False
            
            text_match_start_value = match_from_start_float
            text_match_end_value = match_from_end_float
            no_match = not (match_from_start_bool or match_from_end_bool)
            result_id = f"{value["id"]}R{idx + 1}"
            
            row = [
                result_id,
                correct_file,
                guessed_file,
                filename_match,
                correct_pages,
                guessed_page,
                page_match,
                distance,
                text_match_start_value,
                match_from_start_length,
                text_match_end_value,
                match_from_end_length,
                no_match,
                match_threshold,
                value["difficulty"],
                value["category"],
                value["answer"],
                value["question"],
                document,
                results["ids"][0][idx]
            ]
            all_rows.append(row)
            
    save_data_from_result(all_rows, all_columns, RESULTS_CSV_NAME, RESULTS_EXCEL_NAME)
    

# For query with one question, using embeddings
def query_documents_one_embedding(question, n_results=3):
    results = collection.query(query_embeddings=[question["question_embedding"]], n_results=n_results)
    for idx, document in enumerate(results["documents"][0]):
        distance = results["distances"][0][idx]
        metadata = results["metadatas"][0][idx]  # Include metadata if needed
        print("-" * 60)
        print("-" * 20, f"Result {idx + 1}", "-" * 20)
        print("-" * 60)
        print("Question: ", question["question"])
        print("Answer expected: ", question["answer"])
        print("\nFile from result: ", metadata.get("filename"), " | File from toml: ", question["files"][0]["file"])
        if metadata.get("filename") == question["files"][0]["file"]:
            print("Right File!")
            print("Pages from result", question["files"][0]["page_numbers"], " | Pages from toml: ", metadata.get("page_number"))
            if metadata.get("page_number") in question["files"][0]["page_numbers"]:
                print("Right Pages!")
            else:
                print("Wrong Pages!")
        else:
            print("Wrong File!")
        print("Distance between question and chunk embedding: ", distance)
        print("-" * 30)
        match_strings(document, question["answer"]) # Does not use the returns, just the prints


# For query with one question, using text
# Chroma will first embed each query text with the collection's embedding function, if query_texts is used
def query_documents_text_input(question, n_results=3):
    results = collection.query(query_texts=[question], n_results=n_results)
    # Extract the relevant chunks. Flatten the list of lists
    # results["documents"] is a list of lists, where each sublist corresponds to a document
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    for idx, document in enumerate(results["documents"][0]):
        doc_id = results["ids"][0][idx]
        distance = results["distances"][0][idx]
        metadata = results["metadatas"][0][idx]  # Include metadata if needed
        print("-" * 60)
        print(
            f"Found chunk: ID={doc_id}, Page={metadata.get("page_number")}, Distance={distance}"
        )
        print("-" * 60)
        print(f"Content:\n{document}\n\n---\n")

    return relevant_chunks


# ------------------------------------------------------------------------#
# -----------------------Calling Functions--------------------------------#
# ---------------(Some functions are only meant top be ran once)----------#

# --Step 1
# --This takes all pdf files in PDF_DIRECTORY and parses them.
# --It also tokenizes, Chunks up text, creates embeddings.
# --Then it inserts the embeddings to chromadb, with metadata.
# --Only nedds to be run if you want to add documents to db.
# --Otherwise, run only once. And then comment out
# --------------------------------------------------------------------#
# --Parse, Tokenize, Chunk up, Embedd PDFs and insert into database---#
# --------------------------------------------------------------------#
# process_pdfs_and_insert(PDF_DIRECTORY)

# --OPTIONAL Step 1.5
# --A quick test function, to see if it works.
# --This is just a way to run a fast query from text (not embeddings) on the db.
# --It also returns chunks for later use when generating responses from LLMs.
# --But that's not necessary right now.
# --Should be commented out, unless testing.
# --------------------------------------------------------------#
# --------------Write a (text) question, Run a query------------#
# --------------------------------------------------------------#
# question = "Hur introduceras asylbarnen till det svenska samhället på förskolan?"
# relevant_chunks = query_documents_text_input(question, n_results=RESULTS_PER_QUERY)

# --Step 2
# --This takes all toml files in TOML_DIRECTORY and parses them.
# --Then it adds a key "question_embedding" with the embedding array of the question as value
# --It then saves a new file with "embedded_"-prefix, which is a copy of the old toml, 
# ----but with the embeddings included in the file.
# --Important to keep the "embedded_"-prefix, since other functions use that as a filter,
# ----to choose which file to read.
# --Only needs to be run once, to generate the files.
# --Then just comment out.
# --------------------------------------------------------------#
# -------Write new toml files with embeddings included----------#
# --------------------------------------------------------------#
# add_embeddings_to_toml(TOML_DIRECTORY)


# --Step 3.1
# --This reads the toml files created in Step 2.
# --It returns a dictionary representing the toml files.
# --The key for each entry in the dictionary is the question id from the toml files.
# --This dictionary will be used in queries and to match results
# --Needs to run if you want to use the embeddings from the toml file,
# ----and run a query with the functions 
# ----"query_documents_one_embedding" or 
# ----"query_documents_all_questions"
# --------------------------------------------------------------#
# -------Get the data from toml files, with embedding-----------#
# --------------------------------------------------------------#
# question_dict = get_embedded_questions(TOML_DIRECTORY)



# --Step 3.2, Last step for now.
# --This runs a query on a specified question.
# --The function checks the result of the query, gives distances between question and chunk embedding.
# --Then we compare the metadata of the chunk with the data in the dictionary.
# --We check for filename match, page number match and finally text match.
# --query_documents_one_embedding only checks one embedded question 
# ----from the loaded dictionary and prints results.
# --query_documents_all_embeddings checks all embedded questions
# ----from the loaded dictionary and saves results in csv and xlsx files.
# ----Filenames for saved results are set in the config at the top of this file.
# --Run one everytime you want to query with embeddings.
# --------------------------------------------------------------#
# -------------Run an embedded query from toml files------------#
# --------------------------------------------------------------#
# query_documents_one_embedding(question_dict["PMCSKOLVERKET002"], n_results=RESULTS_PER_QUERY)
# query_documents_all_embeddings(question_dict, n_results=RESULTS_PER_QUERY)



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
