import tiktoken
import os
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------#
# -------------------Config----------------------#
# -----------------------------------------------#
# Load environment variables from .env file
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

PDF_DIRECTORY = "./pdf_data"
COLLECTION_NAME = "docs_collection_norm_all"
PERSIST_DIRECTORY = "docs_storage_norm_all"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
TOKEN_ENCODER = tiktoken.encoding_for_model(EMBEDDING_MODEL_NAME)
MAX_TOKENS = 512
TOML_DIRECTORY = "questions/embedded/"
MATCH_THRESHOLD = 50
RESULTS_PER_QUERY = 5
TOLERANCE = 1
MATCH_THRESHOLD = 50
RESULTS_CSV_NAME = "norm_queries.csv"
RESULTS_EXCEL_NAME = "norm_queries_excel.xlsx"
MIN_ANS_LENGTH = 3
MULTIPROCESSING = True if TOLERANCE > 0 else False
# Set results file names based on multiprocessing and tolerance
if MULTIPROCESSING:
    RESULTS_CSV_NAME = f"results/norm_queries_with_tol{TOLERANCE}.csv"
    RESULTS_EXCEL_NAME = f"results/norm_queries_excel_tol{TOLERANCE}.xlsx"
else:
    RESULTS_CSV_NAME = "results/norm_queries_no_tol.csv"
    RESULTS_EXCEL_NAME = "results/norm_queries_excel_no_tol.xlsx"

FOLDER_PATH = "temp_storage"
OUTPUT_BASE = "./compare_splits_sorted"