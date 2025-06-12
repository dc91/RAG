import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import tomli
from Levenshtein import distance
from Levenshtein import ratio

load_dotenv()

# -----------------------------------------------#
# -------------------Config----------------------#
# -----------------------------------------------#
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TOML_DIRECTORY = "questions/embedded/"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
COLLECTION_NAME = "docs_collection_norm_all"
PERSIST_DIRECTORY = "docs_storage_norm_all"
MATCH_THRESHOLD = 50
RESULTS_PER_QUERY = 5
TOLERANCE = 0

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

def check_shrinking_matches(text_list, chunk, shrink_from_start=False, tolerance=1):
    chunk = chunk.lower()
    text_len = len(text_list)
    chunk_len = len(chunk)
    for i in range(text_len - 3):
        # Determine the current substring based on shrinking direction
        current = text_list[i:] if shrink_from_start else text_list[: text_len - i]
        substring = "".join(current).lower()
        substring_len = len(substring)
        # Use a sliding window over the chunk to compare with the substring
        for j in range(chunk_len - substring_len + 1):
            window = chunk[j:j + substring_len]
            dist = distance(substring, window, score_cutoff=1, score_hint=0)
            ratios = ratio(substring, window)
            # Check if the distance is within the allowed tolerance
            if dist <= tolerance and ratios >= 0.92: # adjusting the ratio threshold to filter out very short answers.
                percent_of_answer_kept = 100.0 * len(current) / text_len
                idx = chunk.find(window)
                print(f"Match within sliding window: \n'{substring}' \n== \n'{window}'")
                print(f"Ratio match within window: {ratios}")
                print(f"Percent of answer kept: {percent_of_answer_kept:.2f}%, {len(substring)}/{text_len} characters kept")
                print(f"Match starts at char position: {idx}")
                print(f"Match ends at char position: {idx + len(substring) - 1}")
                print(f"Match length: {len(substring)}")
                return True, percent_of_answer_kept, substring_len

    return False, 0, 0


def match_strings(chunk_text, answer):
    answer_chars = list(answer.lower())
    print("Full chunk (in lowercase): ", chunk_text.lower())
    print("-" * 30)
    print("[Shrinking from end and matching...]")
    match_from_start_bool = check_shrinking_matches(
        answer_chars, chunk_text, shrink_from_start=False, tolerance=TOLERANCE
    )[0]
    if match_from_start_bool:
        print("(Match from start)")
    else:
        print("(No match from start)")
    print("-" * 30)
    print("[Shrinking from start and matching...]")
    match_from_end_bool = check_shrinking_matches(
        answer_chars, chunk_text, shrink_from_start=True, tolerance=TOLERANCE
    )[0]
    if match_from_end_bool:
        print("(Match from end)")
    else:
        print("(No match from end)")
    return match_from_start_bool, match_from_end_bool

# -----------------------------------------------#
# --------------Query function-------------------#
# -----------------------------------------------#
def query_documents_one_embedding(question, n_results=3):
    results = collection.query(
        query_embeddings=[question["question_embedding"]], n_results=n_results
    )
    for idx, document in enumerate(results["documents"][0]):
        distance = results["distances"][0][idx]
        metadata = results["metadatas"][0][idx]  # Include metadata if needed
        print("\n\n")
        print("-" * 60)
        print("-" * 20, f"Result {idx + 1}", "-" * 20)
        print("-" * 60)
        print("Question: ", question["question"])
        print("Answer expected: ", question["answer"])
        print(
            "\nFile from result: ",
            metadata.get("filename"),
            " | File from toml: ",
            question["files"][0]["file"],
        )
        if metadata.get("filename") == question["files"][0]["file"]:
            print("Right File!")
            print(
                "Pages from result",
                question["files"][0]["page_numbers"],
                " | Pages from toml: ",
                metadata.get("page_number"),
            )
            guessed_page_list = list(map(int, metadata.get("page_number").split(",")))
            page_match = any(page in question["files"][0]["page_numbers"] for page in guessed_page_list) if metadata.get("filename") == question["files"][0]["file"] else False
            if page_match:
                print("Right Pages!")
            else:
                print("Wrong Pages!")
        else:
            print("Wrong File!")
        print("Distance between question and chunk embedding: ", distance)
        print("-" * 30)
        match_strings(
            document, question["answer"]
        )  # Does not use the returns, just the prints


# --------------------------------------------------------------#
# -------Get the data from toml files, with embedding-----------#
# --------------------------------------------------------------#
question_dict = get_embedded_questions(TOML_DIRECTORY)

# --------------------------------------------------------------#
# -------------Run an embedded query from toml files------------#
# --------------------------------------------------------------#
query_documents_one_embedding(question_dict["PAV021"], n_results=RESULTS_PER_QUERY)