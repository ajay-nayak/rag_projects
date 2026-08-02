import os
import re
import hashlib
from typing import List, Dict, Any
import fitz  # PyMuPDF
import pdfplumber


class FinancialPDFParser:
    """Advanced PDF parser for financial reports extracting text, structure, and tables."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

    @staticmethod
    def compute_hash(text: str) -> str:
        """Computes SHA256 hash of text content for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def extract_pages_text(self) -> List[Dict[str, Any]]:
        """Extracts text per page with page numbers."""
        pages_data = []
        try:
            doc = fitz.open(self.file_path)
            for page_idx, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    pages_data.append({
                        "page_number": page_idx + 1,
                        "text": text,
                    })
            doc.close()
        except Exception as e:
            print(f"Error reading PyMuPDF for {self.file_name}: {e}")
        return pages_data

    def extract_tables_markdown(self) -> List[Dict[str, Any]]:
        """Extracts tables per page and converts them to formatted Markdown tables."""
        tables_data = []
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    extracted_tables = page.extract_tables()
                    for t_idx, table in enumerate(extracted_tables):
                        if not table or len(table) < 2:
                            continue
                        
                        # Build Markdown Table
                        headers = [str(cell or "").strip().replace("\n", " ") for cell in table[0]]
                        header_line = "| " + " | ".join(headers) + " |"
                        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
                        
                        rows = []
                        for row in table[1:]:
                            clean_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
                            rows.append("| " + " | ".join(clean_row) + " |")
                        
                        md_table = "\n".join([header_line, separator_line] + rows)
                        
                        tables_data.append({
                            "page_number": page_idx + 1,
                            "table_index": t_idx + 1,
                            "markdown": md_table,
                        })
        except Exception as e:
            print(f"Error extracting tables from {self.file_name}: {e}")
        return tables_data

    def parse(self) -> List[Dict[str, Any]]:
        """Main parse method yielding structured metadata-rich document chunks."""
        chunks = []
        pages = self.extract_pages_text()
        tables = self.extract_tables_markdown()

        # Add text page chunks
        for p in pages:
            chunk_hash = self.compute_hash(p["text"])
            chunks.append({
                "id": f"{self.file_name}_p{p['page_number']}_text_{chunk_hash[:8]}",
                "content": p["text"],
                "metadata": {
                    "source": self.file_name,
                    "page_number": p["page_number"],
                    "is_table": False,
                    "content_hash": chunk_hash,
                }
            })

        # Add table chunks
        for t in tables:
            chunk_hash = self.compute_hash(t["markdown"])
            chunks.append({
                "id": f"{self.file_name}_p{t['page_number']}_tbl{t['table_index']}_{chunk_hash[:8]}",
                "content": f"Financial Table (Page {t['page_number']}):\n" + t["markdown"],
                "metadata": {
                    "source": self.file_name,
                    "page_number": t["page_number"],
                    "is_table": True,
                    "content_hash": chunk_hash,
                }
            })

        return chunks
