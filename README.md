This code was written during the night, look out for misstakes!
```
pip install chromadb openai pymupdf4llm tiktoken tomlkit tomli pandas matplotlib seaborn openpyxl levenshtein tqdm joblib
```
Jupyter files in extra script folder explain a bit more about parts of the code.

checkSplitting.py checks how the splitting looks in normal text and markdown.

parse_embedd_into_db.py is for parsing and chunking files, creating and upserting a collection in chromadb, once you know your files are good.

plotResults.py plots the results. After plot the terminal waits for input, you can press any key at terminal to close the plots.

embedd_toml_questions.py embedds the questions in the toml files and creates new files with embedding included.

query_db_all_questions.py and query_one_question.py, queries the database with embedded questions and return results.

More instructions incoming, but i tried to comment the code, so check it out and read the docs.

chromadb:
https://docs.trychroma.com/docs/overview/getting-started

openai embeddings:
https://platform.openai.com/docs/guides/embeddings
