import os
import fitz
import pathlib
import pymupdf4llm
from concurrent.futures import ProcessPoolExecutor
from config import (
    PDF_DIRECTORY,
    MD_DIRECTORY
)

BY_FILE = False

def process_pdf_by_page(filename):
    if not filename.endswith(".pdf"):
        return
    
    pdf_path = os.path.join(PDF_DIRECTORY, filename)
    doc = fitz.open(pdf_path)
    filename_s = filename[:-4]

    for i, page in enumerate(doc):
        try:
            file_path_md = os.path.join(MD_DIRECTORY, f"{filename_s}_page{i+1}.md")
            md_text = pymupdf4llm.to_markdown(
                os.path.join(PDF_DIRECTORY, filename),
                write_images=False,
                pages=[i],
                filename=filename_s,
            )
            pathlib.Path(file_path_md).write_bytes(md_text.encode())
            print(f"✅ Processed: {filename}")
        except Exception as e:
            print(f"❌ Failed {filename}: {e}")


def process_pdf_by_file(filename):
    for filename in os.listdir(PDF_DIRECTORY):
            if filename.endswith(".pdf"):
                filename_s = filename[:-4]  # Remove '.pdf'
                try:
                    file_path_md = os.path.join(MD_DIRECTORY, filename[:-4] + ".md")

                    md_text = pymupdf4llm.to_markdown(
                        f"./{PDF_DIRECTORY}/{filename}",
                        write_images=False,
                        filename=f"{filename_s}",
                    )
                    pathlib.Path(file_path_md).write_bytes(md_text.encode())
                    print(f"✅ Processed: {filename}")
                except Exception as e:
                    print(f"❌ Failed {filename}: {e}")
                
if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]
    with ProcessPoolExecutor() as executor:
        if BY_FILE:
            executor.map(process_pdf_by_file, pdf_files)
        else:
            executor.map(process_pdf_by_page, pdf_files)