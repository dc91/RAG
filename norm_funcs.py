import re

CONTROL_SPACE_REGEX = re.compile(
    r'[\x00-\x1F\x7F\u00A0\u1680\u180E\u2000-\u200F\u2028\u2029\u202F\u205F\u2060\u2061\u2062\u2063\u2064\uFEFF]'
)

# Norm 1
def normalize_text(input_text):
    # Remove split words at the end of lines
    normalized = re.sub(r"- ?\n", "", input_text.strip())
    # Replace any sequence of whitespace (including newlines) with a single space
    normalized = re.sub(r"\s+", " ", normalized)
    # Don't keep space if end of sentence
    normalized = re.sub(r" +\.\s", ". ", normalized)

    return normalized

# Norm 2
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
    
    return CONTROL_SPACE_REGEX.sub('', merged_text).strip()

# Still Norm 2
def remove_md_stuff(text):
    content = re.sub(r'(?:\n)?#{1,6}|(?:\n)?```(?:.|\n)*?```|(?:\n)?---+|(?:\n)?___+', '', text)

    # Replace spaces with a single space
    content = re.sub(r' +', ' ', content)

    # Remove bold (handles **bold** and __bold__)
    content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', content)

    # Remove italic (handles *italic* and _italic_)
    content = re.sub(r'(\*|_)(.*?)\1', r'\2', content)
        
    return content

