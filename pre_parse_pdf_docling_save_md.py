import os
import fitz
import pathlib
from docling.document_converter import DocumentConverter
from concurrent.futures import ProcessPoolExecutor
from config import (
    PDF_DIRECTORY,
    MD_DIRECTORY
)
converter = DocumentConverter()
def process_pdf_by_page(filename):
    if not filename.endswith(".pdf"):
        return
    
    pdf_path = os.path.join(PDF_DIRECTORY, filename)
    if not os.path.isfile(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    os.makedirs(MD_DIRECTORY, exist_ok=True)
    doc = fitz.open(pdf_path)
    filename_s = filename[:-4]
    result = converter.convert(pdf_path)
    for i, page in enumerate(doc):
        try:
            file_path_md = os.path.join(MD_DIRECTORY, f"{filename_s}_page{i+1}.md")
            md_text = result.document.export_to_markdown(page_no=i+1, image_mode="embedded")
            pathlib.Path(file_path_md).write_bytes(md_text.encode(encoding="utf-8"))
            print(f"✅ Processed: {filename}")
        except Exception as e:
            print(f"❌ Failed {filename}: {e}")

                
if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]
    with ProcessPoolExecutor(max_workers=1) as executor:
        executor.map(process_pdf_by_page, pdf_files)