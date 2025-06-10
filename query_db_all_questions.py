import os
import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import tomli

load_dotenv()

# -----------------------------------------------#
# -------------------Config----------------------#
# -----------------------------------------------#
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TOML_DIRECTORY = "questions/embedded/"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
COLLECTION_NAME = "document_collection_norm_all"
PERSIST_DIRECTORY = "document_storage_norm_all"
MATCH_THRESHOLD = 50
RESULTS_PER_QUERY = 3
RESULTS_CSV_NAME = "results/norm_queries.csv"
RESULTS_EXCEL_NAME = "results/norm_queries_excel.xlsx"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_KEY, model_name=EMBEDDING_MODEL_NAME
)
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=openai_ef
)


# -----------------------------------------------#
# ---------------Helping functions---------------#
# -----------------------------------------------#
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


# Checks the saved answer with the fetched chunk
# If no match, it shrinks by one character and tries again.
# until characters run out or match is found.
# can change range of loop to not shrink until the very last character
# since that is a bad match anyway.
def check_shrinking_matches(
    text_list, chunk, shrink_from_start=False):
    text_len = len(text_list)
    for i in range(text_len - 1):
        current = text_list[i:] if shrink_from_start else text_list[: text_len - i]
        substring = "".join(current)
        if substring in chunk.lower():
            percent_match = 100.0 * len(current) / text_len
            return True, percent_match, len(substring)
    return False, 0, 0


# Only concerns the excel file output
def escape_excel_formulas(val):
    if isinstance(val, str) and val.startswith("="):
        return "'" + val
    return val

# Save results to csv and xlsx files
def save_data_from_result(all_rows, all_columns, csv_name, excel_name):
    df = pd.DataFrame(all_rows, columns=all_columns)
    df.to_csv(csv_name, encoding="utf-8", index=False)

    df = df.map(escape_excel_formulas)
    with pd.ExcelWriter(excel_name) as writer:
        df.to_excel(writer, sheet_name="Test_Query", index=False)


# -----------------------------------------------#
# --------------Query function-------------------#
# -----------------------------------------------#
def query_documents_all_embeddings(question, n_results=3):
    all_columns = [
        "Result_Id",
        "Correct_File",
        "Guessed_File",
        "Filename_Match",
        "Correct_Pages",
        "Guessed_Page",
        "Page_Match",
        "Distance",
        "Text_Match_Start_Percent",
        "Match_Length_Start",
        "Text_Match_End_Percent",
        "Match_Length_End",
        "No_match",
        "Match_Threshold",
        "Difficulty",
        "Category",
        "Expected_answer",
        "Question",
        "Returned_Chunk",
        "Chunk_Id",
    ]
    all_rows = []
    for value in question.values():
        results = collection.query(
            query_embeddings=[value["question_embedding"]], n_results=n_results
        )
        for idx, document in enumerate(results["documents"][0]):  
            # document here refers to chunks, due to chromadb naming
            # so we are looking at the results for returned chunks here.
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

            match_from_start_bool, match_from_start_float, match_from_start_length = (
                check_shrinking_matches(
                    list(value["answer"].lower()), document, shrink_from_start=False
                )
            )
            match_from_end_bool, match_from_end_float, match_from_end_length = (
                check_shrinking_matches(
                    list(value["answer"].lower()), document, shrink_from_start=True
                )
            )
            # We need to figure out what the thershold is, and how to calculate it. This adds both matches.
            # We could use match length somehow as well?
            match_threshold = (
                True
                if (match_from_start_float + match_from_end_float > MATCH_THRESHOLD)
                else False
            )

            text_match_start_value = match_from_start_float
            text_match_end_value = match_from_end_float
            no_match = not (match_from_start_bool or match_from_end_bool)
            result_id = f"{value['id']}R{idx + 1}"

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
                results["ids"][0][idx],
            ]
            all_rows.append(row)
    # Save results to file
    save_data_from_result(all_rows, all_columns, RESULTS_CSV_NAME, RESULTS_EXCEL_NAME)


# --------------------------------------------------------------#
# -------Get the data from toml files, with embedding-----------#
# --------------------------------------------------------------#
question_dict = get_embedded_questions(TOML_DIRECTORY)

# --------------------------------------------------------------#
# -------------Run an embedded query from toml files------------#
# --------------------------------------------------------------#
query_documents_all_embeddings(question_dict, n_results=RESULTS_PER_QUERY)
