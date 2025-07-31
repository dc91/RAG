import os
from openai import OpenAI
import tomli
from Levenshtein import distance
from Levenshtein import ratio
from config import (
    TOML_DIRECTORY_EMBEDDED,
    RESULTS_PER_QUERY,
    TOLERANCE,
    EMBEDDING_MODEL_NAME,
    get_collection,
    get_client
)
from helping_scripts.norm_funcs import normalize_spaces

collection = get_collection()

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
    # results = collection.query(
    #     query_embeddings=[question["question_embedding"]], n_results=n_results
    # )
    results = collection.query(
        query_embeddings=question, n_results=n_results
    )
    for idx, document in enumerate(results["documents"][:]):
        # print(document)
        return document


# --------------------------------------------------------------#
# -------Get the data from toml files, with embedding-----------#
# --------------------------------------------------------------#
# question_dict = get_embedded_questions(TOML_DIRECTORY_EMBEDDED)

# --------------------------------------------------------------#
# -------------Run an embedded query from toml files------------#
# --------------------------------------------------------------#
q = "Hur lång tid måste jag vänta mellan varje behandling med Argos?"
response = get_client().embeddings.create(input=q, model=EMBEDDING_MODEL_NAME)
q_emb = [d.embedding for d in response.data]

relevant_chunks = query_documents_one_embedding(q_emb, n_results=2)
# query_documents_one_embedding(question_dict["DC021"], n_results=RESULTS_PER_QUERY)

# -----------------------------------------------#
# -------------Response from OpenAI--------------#
# -----------------------------------------------#
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    # context = normalize_spaces(context)
    prompt = ("""
        Använd endast det angivna kontextet för att besvara frågan. Lägg inte till information. Om du inte kan svaret på frågan i kontextet,
        säg att du inte vet svaret. Var kortfattad och koncis.
        """
        "\nKontext:\n" + context + "\nFråga:\n" + question
    )
    client_local = OpenAI(base_url="http://192.168.8.3:1234/v1", api_key="not-needed")
    response = client_local.chat.completions.create(
        model="local",
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


answer = generate_response(q, relevant_chunks)
print("Fråga: ", q, "\n\nSvar:", answer.content)