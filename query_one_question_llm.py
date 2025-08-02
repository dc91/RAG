from config import (
    EMBEDDING_MODEL_NAME,
    SYS_PROMPT_FOR_OUTPUT,
    get_collection,
    get_client
)
from helping_scripts.generate_llm_response import generate_response_from_context

collection = get_collection() # Make sure to get the right collection, set in config

# -----------------------------------------------#
# --------------Query function-------------------#
# -----------------------------------------------#
def query_documents_one_embedding(question, n_results=3):
    results = collection.query(
        query_embeddings=question, n_results=n_results
    )
    for idx, document in enumerate(results["documents"]):
        return document


# -----------------------------------------------#
# --------------Question embedding---------------#
# -----------------------------------------------#
question = "Vad innebär det om en webbplats följer riktlinjerna i WCAG 2.0?"
response = get_client().embeddings.create(input=question, model=EMBEDDING_MODEL_NAME)
question_emb = response.data[0].embedding


# -----------------------------------------------#
# --------------Response from LLM----------------#
# -----------------------------------------------#
relevant_chunks = query_documents_one_embedding(question_emb, n_results=2)
answer = generate_response_from_context(question, relevant_chunks, SYS_PROMPT_FOR_OUTPUT)
print("Fråga: ", question, "\n\nSvar:", answer)