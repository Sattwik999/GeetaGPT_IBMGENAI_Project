import os
import re
import json
import fitz  # PyMuPDF
import pandas as pd
from transformers import pipeline

# ✅ Load Hugging Face LLM (no login needed)
generator = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-alpha",  # or "tiiuae/falcon-7b-instruct"
    max_new_tokens=100,
    device_map="auto"
)

# ✅ Extract all text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# ✅ Remove Devanagari/Sanskrit lines
def remove_sanskrit_lines(text):
    lines = text.splitlines()
    english_lines = [line.strip() for line in lines if not re.search(r'[\u0900-\u097F]', line)]
    return "\n".join(english_lines)

# ✅ Extract verses based on format like 2.20 (or fallback to chunks)
def extract_verses(text):
    pattern = r'(\d+\.\d+)\s+([^\n]+(?:\n(?!\d+\.\d+).+)*)'
    matches = re.findall(pattern, text)
    if matches:
        return [(v_id.strip(), v_text.strip().replace('\n', ' ')) for v_id, v_text in matches]
    else:
        # Fallback: paragraph-wise chunks if no clear verses
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
        return [(f"V{i+1}", para) for i, para in enumerate(paras)]

# ✅ Use Hugging Face model to generate a natural question
def generate_question_from_verse(verse_text):
    prompt = f"""You're a spiritual seeker. Read this verse:\n\n"{verse_text}"\n\nWhat thoughtful question might arise in your mind after reading it?"""
    try:
        result = generator(prompt, max_new_tokens=60, do_sample=True)[0]["generated_text"]
        return result.split("\n")[-1].strip()
    except Exception as e:
        print("Error generating question:", e)
        return "What does this verse mean?"

# ✅ Process PDFs in a folder and save enriched dataset
def process_folder_with_questions(pdf_folder, output_path_jsonl, output_path_csv=None):
    all_data = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_folder, filename)
            print(f"📄 Processing: {filename}")
            raw_text = extract_text_from_pdf(file_path)
            english_text = remove_sanskrit_lines(raw_text)
            verses = extract_verses(english_text)

            for verse_id, verse_text in verses:
                print(f"→ Verse {verse_id}")
                question = generate_question_from_verse(verse_text)
                item = {
                    "instruction": "Answer the following question based on the teachings of the Bhagavad Gita.",
                    "input": question,
                    "output": verse_text,
                    "verse_id": verse_id
                }
                all_data.append(item)

    # Save to JSONL
    with open(output_path_jsonl, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ JSONL saved: {output_path_jsonl}")

    # Optional CSV
    if output_path_csv:
        df = pd.DataFrame(all_data)
        df.to_csv(output_path_csv, index=False)
        print(f"✅ CSV saved: {output_path_csv}")

# === MAIN ===
if __name__ == "__main__":
    folder_path = "gita_pdfs"  # Folder where your PDFs are stored
    output_jsonl = "geetgpt_hf_dataset.jsonl"
    output_csv = "geetgpt_hf_dataset.csv"  # Optional
    process_folder_with_questions(folder_path, output_jsonl, output_csv)
