import os
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
# from chromadb import EmbeddingFunction
# from chromadb.utils.embedding_functions import MistralEmbeddingFunction
# from chromadb.utils.embedding_functions import JinaEmbeddingFunction
# from mistralai import Mistral
# import requests
from transformers import AutoTokenizer, AutoModel

#---------------------------------------#
#-----------------KEYS------------------#
#---------------------------------------#
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
COHERE_KEY = os.getenv("CHOHERE_API_KEY")
JINA_KEY = os.getenv("JINA_API_KEY")


#---------------------------------------#
#---------------Directories-------------#
#---------------------------------------#
PDF_DIRECTORY = "pdf_data"
TOML_DIRECTORY_CLEANED = "questions/cleaned"
TOML_DIRECTORY_EMBEDDED = "questions/embedded"
OUTPUT_DIRECTORY_COMPARE_SPLITS = "compare_splits_from_parser"
RESULTS_DIRECTORY = "results_new"
MD_DIRECTORY = "md_data" # If you choose to pre-prase md-files
LOCAL_BASE_URL = "http://192.168.8.3:1234/v1"

#---------------------------------------#
#------------General Settings-----------#
#---------------------------------------#
BASE_NAME = "BASELINE"
USE_OPENAI = True
ADD_LLM_CONTEXT = False
RERANK = False
RERANK_MODEL = "COHERE"# "COHERE", "JINA_API" or "JINA_LOCAL"


#---------------------------------------#
#----------------Parsing----------------#
#---------------------------------------#
PARSE_AS_MD = True
USE_RECURSIVE_SPLIT = False
NORMALIZE_AT_PARSE = False
MAX_TOKENS = 256
OVERLAP =  0


#---------------------------------------#
# --------------Embedding---------------#
#---------------------------------------#
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
# EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3"
# EMBEDDING_MODEL_NAME = "KBLab/sentence-bert-swedish-cased"
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
# EMBEDDING_MODEL_NAME = "mistral-embed"

if USE_OPENAI:
    TOKEN_ENCODER = tiktoken.encoding_for_model(EMBEDDING_MODEL_NAME)
    EMBEDDING_FUNCTION = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_KEY, 
        model_name=EMBEDDING_MODEL_NAME
    )
else:
    # TOKEN_ENCODER = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    # EMBEDDING_FUNCTION = MistralEmbeddingFunction(model=EMBEDDING_MODEL_NAME)
    # EMBEDDING_FUNCTION = JinaEmbeddingFunction(model=EMBEDDING_MODEL_NAME)
    AUTOMODEL_CUSTOM = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    TOKEN_ENCODER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    
def get_client():
    if USE_OPENAI:
        return OpenAI(api_key=OPENAI_KEY)
    else:
        return OpenAI(base_url=LOCAL_BASE_URL, api_key="not-needed")
        # return Mistral(api_key=MISTRAL_KEY)


#---------------------------------------#
#---Automatic naming of VERSION_NAME----#
#---------------------------------------#
NORM_NR = 1 if NORMALIZE_AT_PARSE else 0
BASE_NAME_VERSION = f"{BASE_NAME}_N{NORM_NR}"
if PARSE_AS_MD:
    BASE_NAME_VERSION = f"MD_{BASE_NAME_VERSION}"
COS = False
if COS:
    BASE_NAME_VERSION = f"COS_{BASE_NAME_VERSION}"
    
if OVERLAP > 0:
    VERSION_NAME = f"{BASE_NAME_VERSION}_{MAX_TOKENS}_Chunk_{OVERLAP}_Overlap"
else:
    VERSION_NAME = f"{BASE_NAME_VERSION}_{MAX_TOKENS}_Chunk"


#---------------------------------------#
# -----------ChromaDB Configs-----------#
#---------------------------------------#    
COLLECTION_NAME = f"docs_{VERSION_NAME}_collection"
PERSIST_DIRECTORY = f"docs_{VERSION_NAME}_storage"

def get_collection():  
    chroma_client = chromadb.PersistentClient(path="databases/" + PERSIST_DIRECTORY)
    
    if COS:# If you do not want the standard distance function L2. This uses cosine instead.
        return chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": "cosine", # Cohere models often use cosine space
                    "ef_search": 200,
                    "ef_construction": 200,
                    "max_neighbors": 32,
                    "num_threads": 8
                },
                "embedding_function": EMBEDDING_FUNCTION,
            }
        )
    else:
        return chroma_client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION
        )


#---------------------------------------#
#-------------Results Configs-----------#
#---------------------------------------#
MATCH_THRESHOLD = 35
MIN_ANS_LENGTH = 3
RESULTS_PER_QUERY = 5
FILTER_BY_REG_NR = False # Only applies to kemi data
USE_LLM_ANSWERS = False
LLM_USED = "_GEMMA" if USE_LLM_ANSWERS else "_No_LLM"


if COS:
    DISTANCE = 0.45
else:
    DISTANCE = 0.85
OUTPUT_DIRECTORY_RESULTS = f"{RESULTS_DIRECTORY}/{VERSION_NAME}/"

def get_results_filenames():
    base_dir = os.path.join(RESULTS_DIRECTORY, VERSION_NAME)
    os.makedirs(base_dir, exist_ok=True)
    RESULTS_CSV_NAME = os.path.join(base_dir, f"{VERSION_NAME + LLM_USED}_no_tol.csv")
    RESULTS_EXCEL_NAME = os.path.join(base_dir, f"{VERSION_NAME + LLM_USED}_no_tol.xlsx")

    return RESULTS_CSV_NAME, RESULTS_EXCEL_NAME


#---------------------------------------#
#------------------LLM------------------#
#---------------------------------------#
SYS_PROMPT_FOR_CONTEXT = """
Två texter tillhandahålls: en längre text (originaldokumentet) och ett kortare utdrag som förekommer i den längre texten. 
Kontexten till utdraget ska beskrivas så specifikt och kortfattat som möjligt, utan att nämna uppgiften. 
Fokusera endast på vad utdraget handlar om och dess direkta omgivning. Undvik allmän beskrivning av dokumentet. 
Svaret ska vara på svenska, bestå av högst 200 tecken och innehålla endast en sammanhängande mening eller två. 
Inga rubriker, metainformation eller hänvisningar till uppgiften får förekomma i svaret.
"""

SYS_PROMPT_FOR_OUTPUT = """
Använd endast det angivna kontextet för att besvara frågan. Lägg inte till information. 
Om du inte kan svaret på frågan i kontextet, säg att du inte vet svaret. 
Det första stycket i kontextet har störst chans att innehålla svaret. 
Var kort och koncis och svara alltid på svenska.
"""