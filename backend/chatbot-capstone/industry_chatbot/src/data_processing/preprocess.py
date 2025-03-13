import pdfplumber
import re
import json
import os
from data_processing.question_generator import clean_heading, clean_content, generate_questions

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF while handling empty pages.
    """
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            full_text.append(text)
    return "\n".join(full_text)

def generate_qa_pairs(text):
    """
    Generate Q&A pairs by detecting policy sections and extracting their content.
    """
    qa_pairs = []
    unique_sections = set()

    # Regex to split before headings (two or more words starting with a capital letter)
    section_split_pattern = r'\n(?=(?:[A-Z][A-Za-z0-9&\-/]+(?:\s+|$)){2,})'
    sections = re.split(section_split_pattern, text)

    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        parts = section.split('\n', 1)
        heading = parts[0].strip()
        content = parts[1].strip() if len(parts) > 1 else ""

        # Clean heading and content
        heading = clean_heading(heading)
        content = clean_content(content)

        # Skip weak sections (duplicates, metadata, very short content)
        if not content or len(content) < 20 or content.lower().startswith("date :") or heading in unique_sections:
            continue

        unique_sections.add(heading)  # Track unique headings

        questions = generate_questions(heading)

        for q in questions:
            qa_pairs.append({
                "question": q,
                "answer": content,
                "metadata": {
                    "section": heading,
                    "source": "IT Policy Document"
                }
            })

    return qa_pairs

def process_pdf(pdf_path, processed_text_file, output_json_file):
    """
    Process the PDF, save cleaned text, generate Q&A pairs, and save them as a JSON file.
    """
    text = extract_text_from_pdf(pdf_path)

    # Save cleaned extracted text to a file
    with open(processed_text_file, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"✅ Extracted text saved to {processed_text_file}")

    # Generate Q&A pairs
    qa_pairs = generate_qa_pairs(text)

    # Save Q&A pairs as JSON
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {len(qa_pairs)} Q&A pairs. Saved to {output_json_file}")

# Example usage: Adjust file paths as needed.
if __name__ == "__main__":
    pdf_path = r'C:\Users\moksh\classroom\chatbot_deepseek\industry_chatbot\data\raw\pdf 2.pdf'
    processed_text_file = r'C:\Users\moksh\classroom\chatbot_deepseek\industry_chatbot\data\processed\cleaned_text.txt'
    output_json_file = r'C:\Users\moksh\classroom\chatbot_deepseek\industry_chatbot\data\knowledge_base\qa_pairs.json'

    # Ensure directories exist
    os.makedirs(os.path.dirname(processed_text_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)

    process_pdf(pdf_path, processed_text_file, output_json_file)
