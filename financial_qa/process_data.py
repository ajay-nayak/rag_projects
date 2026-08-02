import os
import re
import json
import pdfplumber
import pymupdf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# === Step 1: Extract and Structure PDF Data ===
def extract_text_from_pdf(file_path):
    """Extracts raw text from a PDF."""
    text = ""
    try:
        with pymupdf.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def extract_tables_from_pdf(file_path):
    """Extracts tables from a PDF and converts them to formatted text."""
    tables_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_table()
                if tables:
                    tables_text += "\n".join([" | ".join(row) for row in tables if row]) + "\n\n"
    except Exception as e:
        print(f"Error extracting tables from {file_path}: {e}")
    return tables_text

def detect_sections_and_subsections(text):
    """Detects sections and subsections dynamically."""
    sections = {}
    current_section = "General Information"
    current_subsection = None
    sections[current_section] = {}

    section_pattern = re.compile(r"^(?:\d+\.\s*)?[A-Z][A-Za-z\s\-]+$")
    subsection_pattern = re.compile(r"^(?:\d+\.\d+\s*)?[A-Z][A-Za-z\s\-]+$")

    lines = text.split("\n")
    for line in lines:
        line = line.strip()

        if re.match(section_pattern, line):
            current_section = line.strip()
            sections[current_section] = {}
            current_subsection = None

        elif re.match(subsection_pattern, line):
            current_subsection = line.strip()
            sections[current_section][current_subsection] = ""

        else:
            if current_subsection:
                sections[current_section][current_subsection] += line + " "
            else:
                sections[current_section]["General"] = sections[current_section].get("General", "") + line + " "

    return sections

def process_reports_folder(folder_path):
    """Reads PDFs, extracts sections, subsections, and tables."""
    all_reports = {}

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing: {file_path}")

            pdf_text = extract_text_from_pdf(file_path)
            pdf_tables = extract_tables_from_pdf(file_path)

            full_text = pdf_text + "\n\nExtracted Tables:\n" + pdf_tables
            structured_data = detect_sections_and_subsections(full_text)

            all_reports[file_name] = structured_data

    return all_reports

# Save structured data
def save_structured_reports_to_json(structured_reports, output_file):
    """Saves structured reports to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(structured_reports, f, indent=4)

# Process and save reports
reports_folder = "reports"
structured_reports = process_reports_folder(reports_folder)
save_structured_reports_to_json(structured_reports, "structured_reports.json")
