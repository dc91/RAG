import os
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ------------Directories------------#
PDF_DIRECTORY = "pdf_data"
TOML_DIRECTORY_CLEANED = "questions/cleaned"
TOML_DIRECTORY_EMBEDDED = "questions/embedded"
OUTPUT_DIRECTORY_COMPARE_SPLITS = "compare_splits_from_parser"
RESULTS_DIRECTORY = "results"

# ----OpenAI and ChromaDB Configs----#
MAX_TOKENS = 384
OVERLAP = 5

if OVERLAP > 0:
    VERSION_NAME = f"BASELINE_{MAX_TOKENS}_Chunk_{OVERLAP}_Overlap"
else:
    VERSION_NAME = f"BASELINE_{MAX_TOKENS}_Chunk"
    
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = f"docs_{VERSION_NAME}_collection"
PERSIST_DIRECTORY = f"docs_{VERSION_NAME}_storage"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
TOKEN_ENCODER = tiktoken.encoding_for_model(EMBEDDING_MODEL_NAME)


def get_client():
    return OpenAI(api_key=OPENAI_KEY)

def get_collection():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_KEY, 
        model_name=EMBEDDING_MODEL_NAME
    )
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=openai_ef
    )

# ----------Results Configs----------#
MATCH_THRESHOLD = 30
MIN_ANS_LENGTH = 3
RESULTS_PER_QUERY = 5
TOLERANCE = 0
MULTIPROCESSING = True if TOLERANCE > 0 else False

def get_results_filenames():
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
    if MULTIPROCESSING:
        RESULTS_CSV_NAME = f"{RESULTS_DIRECTORY}/{VERSION_NAME}{TOLERANCE}_tol.csv"
        RESULTS_EXCEL_NAME = f"{RESULTS_DIRECTORY}/{VERSION_NAME}{TOLERANCE}_tol.xlsx"
    else:
        RESULTS_CSV_NAME = f"{RESULTS_DIRECTORY}/{VERSION_NAME}_no_tol.csv"
        RESULTS_EXCEL_NAME = f"{RESULTS_DIRECTORY}/{VERSION_NAME}_no_tol.xlsx"
    return RESULTS_CSV_NAME, RESULTS_EXCEL_NAME