"""
src/data_loader.py — PDF → chonkie chunks → Documents
"""
import re
from typing import List
from dataclasses import dataclass
from pathlib import Path

import fitz
from tqdm.auto import tqdm
from langchain_core.documents import Document
from chonkie import RecursiveChunker

from src.config import get_config

@dataclass
class PageData:
    book: str
    page_num: int
    text: str

class PDFChunker:
    def __init__(self):
        config = get_config()
        tp = config.text_processing
        # ✅ ИСПРАВЛЕНО: RecursiveChunker принимает ТОЛЬКО chunk_size!
        self.chunker = RecursiveChunker(
            chunk_size=tp.chunk_size  # БЕЗ chunk_overlap!
        )
        self.overlap_size = tp.overlap_size  # Сохраняем для информации
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Очистка текста"""
        # 'пациент-вол' → 'пациент вол' (пробел после дефиса) ✅
        text = re.sub(r'(\w+)-\s*(\w+)', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text.replace('\n', ' ').replace('\xa0', ' '))
        text = re.sub(r'—\s*Примеч\.\s*пер\.', '', text)
        return text.strip()
    
    def load_and_chunk_pdf(self, file_path: str) -> List[Document]:
        """1 PDF → chunks"""
        book = Path(file_path).stem
        doc = fitz.open(file_path)
        
        page_texts = []
        for page_num, page in tqdm(enumerate(doc), total=len(doc), desc=book):
            text = self.clean_text(page.get_text("text"))
            if len(text) < 100: 
                continue
            page_texts.append(PageData(book, page_num, text))
        
        doc.close()
        print(f"📄 {book}: {len(page_texts)} страниц")
        
        # Chunking
        chunks = []
        chunk_id = 0
        for page in page_texts:
            page_chunks = self.chunker(page.text)  # str → list[str]
            for i, chunk_text in enumerate(page_chunks):
                chunks.append(Document(
                    page_content=chunk_text.text,
                    metadata={
                        "book": page.book,
                        "page": page.page_num,
                        "chunk_index": i,
                        "chunk_id": chunk_id,
                        "char_count": len(chunk_text.text),
                    }
                ))
                chunk_id += 1
        
        print(f"✅ {len(chunks)} чанков")
        return chunks
    
    def load_multiple(self, pdf_paths: List[str]) -> List[Document]:
        """Несколько PDF → все chunks"""
        all_docs = []
        for path in pdf_paths:
            all_docs.extend(self.load_and_chunk_pdf(path))
        return all_docs

def get_pdf_chunker():
    return PDFChunker()
