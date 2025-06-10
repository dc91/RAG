import os
from openai import OpenAI
from tomlkit import (
    parse,
    dumps,
)
from dotenv import load_dotenv
load_dotenv()


# -----------------------------------------------#
# -------------------Config----------------------#
# -----------------------------------------------#
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TOML_DIRECTORY = "questions/cleaned/"
client = OpenAI(api_key=OPENAI_KEY)


# -----------------------------------------------#
# ---------------Embedding Question--------------#
# -----------------------------------------------#
def get_embedding(question):
    return client.embeddings.create(input=question, model=EMBEDDING_MODEL_NAME).data[0].embedding

def add_embeddings_to_toml(toml_dir):
    for filename in os.listdir(toml_dir):
        if filename.endswith(".toml"):
            with open(toml_dir + filename, "r", encoding="utf-8") as f:
                toml_f = parse(f.read())
            for question in toml_f["questions"]:
                question["question_embedding"] = get_embedding(question["question"])
            with open(toml_dir + "embedded_" + filename, "w", encoding="utf-8") as f:
                f.write(dumps(toml_f))

# --This takes all toml files in TOML_DIRECTORY_INPUT and parses them.
# --Then it adds a key "question_embedding" with the embedding array of the question as value.
# --It then saves a new file. The file will have an "embedded_"-prefix, 
# ----which is a copy of the old toml, but with the embeddings included in the file.
# --Important to keep the "embedded_"-prefix, since other functions use that as a filter,
# ----to choose which file to read.
# --------------------------------------------------------------#
# -------Write new toml files with embeddings included----------#
# --------------------------------------------------------------#
add_embeddings_to_toml(TOML_DIRECTORY)