# from concurrent.futures import ProcessPoolExecutor
# from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pymupdf4llm
import os
import pathlib
import re
# import fitz

MD_DIRECTORY = "md_data"
PDF_DIRECTORY = "pdf_data"
out_base = MD_DIRECTORY
img_base = MD_DIRECTORY + "/images"
md_base = MD_DIRECTORY + "/md"
txt_base = MD_DIRECTORY + "/txt"
os.makedirs(out_base, exist_ok=True)
os.makedirs(img_base, exist_ok=True)
os.makedirs(md_base, exist_ok=True)
os.makedirs(txt_base, exist_ok=True)
MIN_CHUNK = 128
MAX_CHUNK = 512
CONTROL_SPACE_REGEX = re.compile(
    r'[\x00-\x1F\x7F\u00A0\u1680\u180E\u2000-\u200F\u2028\u2029\u202F\u205F\u2060\u2061\u2062\u2063\u2064\uFEFF]'
)

# def process_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text_and_pagenumber = []  # List [(page_number, page_text)]
    
#     for i, page in enumerate(doc):
#         file_path_md = os.path.join(md_base, f"{doc.name}_page{i+1}.md")
#         text = pymupdf4llm.to_markdown(
#                     pdf_path,
#                     write_images=False,
#                     pages=[i],
#                     filename=f"{doc.name}_page{i+1}.md"
#                 )
#         pathlib.Path(file_path_md).write_bytes(text.encode())
#         # text = page.get_text(sort=True) # sort helps keep the right reading order in the page
#         # if text.strip():  # Skip empty pages
#         #     norm_text = normalize_text(text)
#         #     text_and_pagenumber.append((i + 1, norm_text + " "))
#     doc.close()
#     return text_and_pagenumber

# def process_pdf(filename):
#     if not filename.endswith(".pdf"):
#         return
    
#     pdf_path = os.path.join(PDF_DIRECTORY, filename)
#     doc = fitz.open(pdf_path)
#     filename_s = filename[:-4]

#     for i, page in enumerate(doc):
#         try:
#             file_path_md = os.path.join(md_base, f"{filename_s}_page{i+1}.md")
#             md_text = pymupdf4llm.to_markdown(
#                 os.path.join(PDF_DIRECTORY, filename),
#                 write_images=False,
#                 pages=[i],
#                 filename=filename_s,
#             )
#             pathlib.Path(file_path_md).write_bytes(md_text.encode())
#             print(f"✅ Processed: {filename}")
#         except Exception as e:
#             print(f"❌ Failed {filename}: {e}")
        
def process_pdf(filename):
    for filename in os.listdir(PDF_DIRECTORY):
            if filename.endswith(".pdf"):
                filename_s = filename[:-4]  # Remove '.pdf'
                # pdf_path = os.path.join(pdf_directory, filename)
                file_path_md = os.path.join(md_base, filename[:-4] + ".md")

                md_text = pymupdf4llm.to_markdown(
                    f"./{PDF_DIRECTORY}/{filename}",
                    write_images=False,
                    filename=f"{filename_s}",
                )
                pathlib.Path(file_path_md).write_bytes(md_text.encode())

def clean_md_text(text):
    # Split text into lines
    text = re.sub(r"-[\u00AD\u200B\u200C\u200D\u200E\u200F]*\s*\n[\u00AD\u200B\u200C\u200D\u200E\u200F]*\s*", "", text)
    text = re.sub(r"[\u00AD\u200B\u200C\u200D\u200E\u200F]\s*", "", text)
    
    lines = text.split('\n')
    
    cleaned_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        # Skip lines that only contain a number (e.g., page numbers)
        if re.fullmatch(r'\s*\d+\s*', line):
            continue
        # Skip empty lines
        if not stripped_line:
            continue
        cleaned_lines.append(line)
    
    # Join all lines into one paragraph-like text
    merged_text = " ".join(cleaned_lines)
    # Normalize whitespace
    merged_text = re.sub(r"\s+", " ", merged_text)
    # Clean up space before punctuation
    # merged_text = re.sub(r" +\.\s", ". ", merged_text)
    
    return CONTROL_SPACE_REGEX.sub('', merged_text).strip()

def remove_md_stuff(text):
    content = re.sub(r'(?:\n)?#{1,6}|(?:\n)?```(?:.|\n)*?```|(?:\n)?---+|(?:\n)?___+', '', text)

    # Replace spaces with a single space
    content = re.sub(r' +', ' ', content)

    # Remove bold (handles **bold** and __bold__)
    content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', content)

    # Remove italic (handles *italic* and _italic_)
    content = re.sub(r'(\*|_)(.*?)\1', r'\2', content)
        
    return content

def get_sentence_end(paragraph):
    sentence_end = max(
        (m.end() for m in re.finditer(r'(?<=[.!?])\s', paragraph[:MAX_CHUNK])),
        default=None
    )
    if not sentence_end:
        sentence_end = paragraph[:MAX_CHUNK].rfind('\n')
    if sentence_end <= 0:
        sentence_end = paragraph[:MAX_CHUNK].rfind(' ')
    if sentence_end <= 0:
        sentence_end = MAX_CHUNK
    return sentence_end

def split_large_paragraph(paragraph):
    chunks = []
    while len(paragraph) > MAX_CHUNK:
        sentence_end = get_sentence_end(paragraph)
        chunk = paragraph[:sentence_end].strip()
        chunks.append(chunk)
        paragraph = paragraph[sentence_end:].strip()
    if paragraph:
        chunks.append(paragraph)
    return chunks

def is_title_like(paragraph):
    words = paragraph.strip().split()
    return len(paragraph) < 50 and len(words) <= 6

def is_page_number_like(paragraph):
    stripped = paragraph.strip()
    return stripped.isdigit() and 1 <= int(stripped) <= 999

def split_into_paragraphs(text):
    lines = text.splitlines()
    paragraphs = []
    buffer = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("|"):
            # Table row
            buffer.append(line)
            in_table = True
        elif in_table and not stripped:
            # Blank line ends the table
            paragraphs.append("\n".join(buffer).strip())
            buffer = []
            in_table = False
        elif in_table:
            # Still in table
            buffer.append(line)
        elif not stripped:
            # Blank line ends current paragraph
            if buffer:
                paragraphs.append("\n".join(buffer).strip())
                buffer = []
        else:
            # Normal paragraph line
            if is_page_number_like(line):
                continue
            buffer.append(line)

    # Add any trailing content
    if buffer:
        paragraphs.append("\n".join(buffer).strip())

    return [p for p in paragraphs if p]

def para_split(text):
    full_text = []
    growing_chunk = ""
    title_buffer = ""
    paragraphs = split_into_paragraphs(text)

    for paragraph in paragraphs:
        para_len = len(paragraph)

        if paragraph.startswith("|"):
            # Always flush before and after a table block
            if growing_chunk.strip():
                full_text.append(growing_chunk.strip())
                growing_chunk = ""
            full_text.append(paragraph)
            continue

        if is_title_like(paragraph):
            title_buffer = paragraph
            continue
        
        

        if title_buffer:
            paragraph = title_buffer + " " + paragraph
            title_buffer = ""

        if para_len < 50:
            growing_chunk += paragraph + " "
        elif para_len < MIN_CHUNK and len(growing_chunk) + para_len < MAX_CHUNK:
            growing_chunk += paragraph + " "
        elif para_len < MIN_CHUNK and len(growing_chunk) + para_len > MAX_CHUNK:
            full_text.append(growing_chunk.strip())
            growing_chunk = paragraph + " "
        else:
            if growing_chunk.strip():
                full_text.append(growing_chunk.strip())
                growing_chunk = ""

            if para_len > MAX_CHUNK:
                full_text.extend(split_large_paragraph(paragraph))
            else:
                full_text.append(paragraph)

    if title_buffer:
        growing_chunk += title_buffer + " "

    if growing_chunk.strip():
        full_text.append(growing_chunk.strip())

    merged_chunks = []
    buffer = ""

    for chunk in full_text:
        if len(chunk) < MIN_CHUNK:
            buffer += " " + chunk
        else:
            if buffer.strip():
                merged_chunks.append(buffer.strip())
                buffer = ""
            merged_chunks.append(chunk)

    if buffer.strip():
        if merged_chunks:
            merged_chunks[-1] += " " + buffer.strip()
        else:
            merged_chunks.append(buffer.strip())

    # === Debugging Info ===
    # chunk_lengths = [len(chunk) for chunk in merged_chunks]
    # min_len = min(chunk_lengths)
    # max_len = max(chunk_lengths)
    # min_idx = chunk_lengths.index(min_len)
    # max_idx = chunk_lengths.index(max_len)

    # if min_len < 100 or max_len > 1000:
        # print(f"Smallest chunk: Index {min_idx}, Length {min_len}")
        # print(f"Preview: {merged_chunks[min_idx][:100]!r}")
        # print(f"Largest chunk: Index {max_idx}, Length {max_len}")
        # print(f"Preview: {merged_chunks[max_idx][:100]!r}")
    
    return merged_chunks


def merge_short_docs(docs, min_length=100):
    if not docs:
        return []

    merged_docs = [docs[0]]  # start with the first doc

    for doc in docs[1:]:
        if len(merged_docs[-1]) < min_length:
            # If the last one is too short, append the current doc to it
            merged_docs[-1] += ' ' + doc
        else:
            # Otherwise, start a new entry
            merged_docs.append(doc)

    return merged_docs

# def normalize_text(input_text):
#     # Remove split words at the end of lines
#     normalized = re.sub(r"- ?\n", "", input_text.strip())
#     # Replace any sequence of whitespace (including newlines) with a single space
#     normalized = re.sub(r"\s+", " ", normalized)
#     # Don't keep space if end of sentence
#     normalized = re.sub(r" +\.\s", ". ", normalized)

#     return normalized

if __name__ == "__main__":
    # pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]
    # with ProcessPoolExecutor() as executor:
    #     executor.map(process_pdf, pdf_files)
        
    md_files = list(pathlib.Path(md_base).rglob("*.md"))
    # splitter = MarkdownTextSplitter(chunk_size=MIN_CHUNK, chunk_overlap=0)
    # splitter = RecursiveCharacterTextSplitter(
    #     # Set a really small chunk size, just to show.
    #     chunk_size=512,
    #     chunk_overlap=0,
    #     length_function=len,
    #     separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    # )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-small",
        chunk_size=128,
        chunk_overlap=0,
    )
    ShortCount=0
    for md_file in md_files:
        # Read the .md file
        with md_file.open("r", encoding="utf-8") as f:
            markdown_text = f.read()

        # markdown_text = remove_md_stuff(markdown_text)
        # Split into chunks
        # docs = para_split(markdown_text)
        docs = splitter.split_text(markdown_text)
        # pre = len(docs)
        
        docs = merge_short_docs(docs, min_length=100)
        # docs = merge_short_docs(docs, min_length=200)

        # post = len(docs)
        # if not pre == post:
        #     print("DIFF: ", pre, " | " ,post)
        # Get the relative path from md_base
        relative_path = md_file.relative_to(md_base)

        # Build the new path under txt_base with .txt extension
        txt_file_path = txt_base / relative_path.with_suffix(".txt")

        # Ensure the parent directory exists
        txt_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write chunks to the new .txt file
        with txt_file_path.open("w", encoding="utf-8") as f:
            # f.write(f"Chunk: 1\n{clean_md_text(markdown_text)}\n\n")
            for i, chunk in enumerate(docs):
                if len(chunk) < 100:
                    ShortCount+=1
                chunk = remove_md_stuff(chunk)
                # chunk = clean_md_text(chunk)
                # if not is_page_number_like(chunk):
                #     f.write(f"Chunk: {i+1}\n{chunk}\n\n")
                f.write(f"Chunk: {i+1}\n{chunk}\n\n")
    if ShortCount > 0:
        print(ShortCount)