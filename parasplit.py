import re
from config import (
    MAX_TOKENS
)

MIN_CHUNK = 64

def get_sentence_end(paragraph):
    sentence_end = max(
        (m.end() for m in re.finditer(r'(?<=[.!?])\s', paragraph[:MAX_TOKENS])),
        default=None
    )
    if not sentence_end:
        sentence_end = paragraph[:MAX_TOKENS].rfind('\n')
    if sentence_end <= 0:
        sentence_end = paragraph[:MAX_TOKENS].rfind(' ')
    if sentence_end <= 0:
        sentence_end = MAX_TOKENS
    return sentence_end

def split_large_paragraph(paragraph):
    chunks = []
    while len(paragraph) > MAX_TOKENS:
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
        elif para_len < MIN_CHUNK and len(growing_chunk) + para_len < MAX_TOKENS:
            growing_chunk += paragraph + " "
        elif para_len < MIN_CHUNK and len(growing_chunk) + para_len > MAX_TOKENS:
            full_text.append(growing_chunk.strip())
            growing_chunk = paragraph + " "
        else:
            if growing_chunk.strip():
                full_text.append(growing_chunk.strip())
                growing_chunk = ""

            if para_len > MAX_TOKENS:
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