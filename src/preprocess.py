import os
import sys
# This MUST be the very first thing that happens
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pathlib import Path



headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

def preprocess(data_folder):
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    base_path = Path(data_folder)
    md_splits = {}
    for md_file in base_path.iterdir():
        # print(md_file)
        if md_file.suffix == ".md":  # Ensure we only read markdown files
            # pass 1
            content = md_file.read_text(encoding="utf-8") # Read the file content
            md_header_splits = markdown_splitter.split_text(content)
            for doc in md_header_splits:

                header_text = (
                    doc.metadata.get("Header 3") or
                    doc.metadata.get("Header 2") or 
                    doc.metadata.get("Header 1") or 
                    "General Section"
                )
                doc.metadata["article"] = header_text
                doc.metadata["source"] = md_file.name # Tag the document with its filename
                doc.metadata["lang"] = "pl" if "polish" in md_file.name.lower() else "en" # Auto-tag language

            #pass 2
            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=150,
                                separators=["\n\n", "\n", ". ", " ", ""] 
                            )
            md_final_docs = text_splitter.split_documents(md_header_splits)
            md_splits[md_file.name] = md_final_docs # Use the filename as the key


    #print(md_splits["english_document_ai_act.md"])
    return md_splits

if __name__ == "__main__":
    preprocess(data_folder="./data/markdown")