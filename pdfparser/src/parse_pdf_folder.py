import os
import re
from tqdm import tqdm
from PyPDF2 import PdfReader
import argparse

def remove_first_line(text):
    """Функция для удаления первой строки текста."""
    return '\n'.join(text.split('\n')[1:])

def extract_text_from_pdf(pdf_path, output_dir):
    """Обрабатывает один PDF-файл, извлекая текст с каждой страницы."""
    reader = PdfReader(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]  # Имя файла без расширения

    file_output_dir = os.path.join(output_dir, pdf_name)
    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)

    for page_num in tqdm(range(len(reader.pages)), desc=f"Processing {pdf_name}"):
        page = reader.pages[page_num]
        
        text = page.extract_text()

        if not text:
            continue

        # Очистка текста
        text = remove_first_line(text)
        text = re.sub(r"Рис.\s?\d+.\d+.", "", text, flags=re.IGNORECASE)
        
        output_file = os.path.join(file_output_dir, f"{pdf_name}_page_{page_num}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Done parsing {pdf_name}. Results in {file_output_dir}")

def process_pdf_folder(folder_path, output_dir):
    """Обрабатывает все PDF-файлы из указанной папки."""
    if not os.path.exists(folder_path):
        print(f"Input folder {folder_path} does not exist.")
        return

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {folder_path}.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        extract_text_from_pdf(pdf_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a folder of PDF files into individual cleaned-up pages.")
    parser.add_argument("folder_path", metavar='FOLDER_PATH', type=str, 
                        help="Path to the folder containing PDF files.")
    parser.add_argument("--output_dir", metavar='OUTPUT_DIR', type=str, default="output",
                        help="Directory where the output text files will be saved. Default is 'output'.")

    args = parser.parse_args()

    process_pdf_folder(args.folder_path, args.output_dir)