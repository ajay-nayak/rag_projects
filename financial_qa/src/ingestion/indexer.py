import os
import json
import hashlib
from typing import List, Dict, Any, Tuple
from pathlib import Path

from src.utils.config import REPORTS_DIR, INDEX_CACHE_DIR
from src.ingestion.parser import FinancialPDFParser

REGISTRY_FILE = INDEX_CACHE_DIR / "indexing_registry.json"


class IncrementalIndexer:
    """Tracks report files and content hashes to index only new or updated PDFs."""

    def __init__(self, reports_dir: Path = REPORTS_DIR):
        self.reports_dir = reports_dir
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, str]:
        if REGISTRY_FILE.exists():
            try:
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not read registry: {e}")
        return {}

    def _save_registry(self):
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)


    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Calculate SHA256 of the PDF file itself."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_pending_and_modified_files(self) -> Tuple[List[str], List[str]]:
        """Identifies new/modified PDF files needing indexing and removed files."""
        current_files = {}
        for fname in os.listdir(self.reports_dir):
            if fname.lower().endswith(".pdf"):
                fpath = os.path.join(self.reports_dir, fname)
                current_files[fname] = self.get_file_hash(fpath)

        to_process = []
        for fname, fhash in current_files.items():
            if fname not in self.registry or self.registry[fname] != fhash:
                to_process.append(fname)

        to_remove = [fname for fname in self.registry if fname not in current_files]

        return to_process, to_remove

    def parse_new_reports(self) -> List[Dict[str, Any]]:
        """Parses pending PDFs into chunks and updates registry."""
        to_process, to_remove = self.get_pending_and_modified_files()
        all_new_chunks = []

        for fname in to_process:
            fpath = os.path.join(self.reports_dir, fname)
            print(f"Incremental Indexer: Parsing new/updated report -> {fname}")
            parser = FinancialPDFParser(fpath)
            chunks = parser.parse()
            all_new_chunks.extend(chunks)
            
            # Update registry
            self.registry[fname] = self.get_file_hash(fpath)

        for fname in to_remove:
            print(f"Incremental Indexer: Removing deleted report record -> {fname}")
            del self.registry[fname]

        if to_process or to_remove:
            self._save_registry()

        return all_new_chunks
