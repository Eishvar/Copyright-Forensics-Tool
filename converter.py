import pdfplumber
import json
import os
from pathlib import Path

def convert_pdfs_to_individual_jsons(input_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Path to your PDF files
    path = Path(input_dir)
    pdf_files = list(path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting conversion...")

    for pdf_file in pdf_files:
        json_data = []
        file_name = pdf_file.name
        output_file_name = pdf_file.stem + ".json"
        output_path = os.path.join(output_dir, output_file_name)

        print(f"--- Processing: {file_name} ---")

        try:
            with pdfplumber.open(pdf_file) as pdf:
                # Loop through every page (accurate for 100+ page docs)
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    
                    if text:
                        # Constructing a structured format for your RAG system
                        page_entry = {
                            "source": file_name,
                            "page_number": i + 1,
                            "content": text.strip()
                        }
                        json_data.append(page_entry)
                    
                    if (i + 1) % 20 == 0:
                        print(f"  > Processed {i + 1} pages...")

            # Save as a separate JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            
            print(f"✅ Successfully saved: {output_file_name}\n")

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

if __name__ == "__main__":
    # Settings
    INPUT_FOLDER = "data"         # Where your 7 PDFs are
    OUTPUT_FOLDER = "json_data"   # Where you want the 7 JSONs to go
    
    convert_pdfs_to_individual_jsons(INPUT_FOLDER, OUTPUT_FOLDER)
    print("Conversion process complete.")