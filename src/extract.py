import pymupdf4llm
import pathlib
from pathlib import Path

def extract(data_folder):
    base_path = Path(data_folder)
    output_dir = Path('./data/markdown')
    for file_path in base_path.iterdir():
        if file_path.is_file():
            print(f"Extracting {file_path.name} ..")

            md_text = pymupdf4llm.to_markdown(file_path)
            target_path = output_dir / file_path.with_suffix(".md").name

            target_path.write_bytes(md_text.encode())
    return

if __name__ == "__main__":
    extract("./data/")