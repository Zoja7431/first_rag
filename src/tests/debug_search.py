"""
src/debug_search.py — ручной тест поиска по Qdrant
Запуск:
    python -m src.debug_search
"""

from src.vector_store import get_vector_store_manager

def run_search(query: str, k: int = 3) -> None:
    vs = get_vector_store_manager()
    results = vs.search(query, k=k)

    print(f"\n🔍 Query: {query}")
    print(f"Found {len(results)} results")
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        print("\n" + "-" * 80)
        print(f"Result {i}")
        print(f"  Book: {meta.get('book')}")
        print(f"  Page: {meta.get('page')}")
        print(f"  Chunk id: {meta.get('chunk_id')}")
        print(f"  Char count: {meta.get('char_count')}")
        print(f"  Text: {doc.page_content[:400]}...")

if __name__ == "__main__":
    run_search("как понимать язык телодвижений", k=3)
    run_search("игры, в которые играют люди", k=3)
