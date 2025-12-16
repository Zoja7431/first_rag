"""
src/index_documents.py — индексация PDF в Qdrant
Запускать из корня проекта:
    python -m src.index_documents
или
    python src/index_documents.py
"""

from pathlib import Path

from src.config import get_config
from src.data_loader import get_pdf_chunker
from src.vector_store import get_vector_store_manager


def index_single_book(book_stem: str) -> int:
    """
    Индексировать одну книгу (по имени без .pdf, как в config.data.books).
    Возвращает количество загруженных чанков.
    """
    config = get_config()
    chunker = get_pdf_chunker()
    vs = get_vector_store_manager()

    pdf_dir = Path(config.data.pdf_path)
    pdf_path = pdf_dir / f"{book_stem}.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n📚 Indexing book: {pdf_path.name}")
    chunks = chunker.load_and_chunk_pdf(str(pdf_path))
    print(f"✅ Got {len(chunks)} chunks")

    ids = vs.add_documents(chunks)
    print(f"✅ Uploaded {len(ids)} chunks to Qdrant")

    stats = vs.get_collection_stats()
    print(f"📊 Collection '{stats['collection_name']}' now has {stats['points_count']} points")

    return len(ids)


def index_all_books() -> None:
    """
    Пройтись по всем книгам из config.data.books и загрузить в Qdrant.
    """
    config = get_config()
    total_chunks = 0

    for book_stem in config.data.books:
        try:
            count = index_single_book(book_stem)
            total_chunks += count
        except FileNotFoundError as e:
            print(f"⚠️ Skipping '{book_stem}': {e}")

    print(f"\n🎯 Total indexed chunks: {total_chunks}")


if __name__ == "__main__":
    # Можно выбрать: индексировать все книги или только одну
    index_all_books()
    # или, для отладки:
    # index_single_book("Allana-Piza-YAzyk-telodvizhenij")
