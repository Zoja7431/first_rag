import fitz
import re
from spacy.lang.ru import Russian
from tqdm.auto import tqdm
from src.config import config

nlp = Russian()
nlp.add_pipe("sentencizer")

def text_formatter(text: str) -> str:
    # Ваш код форматирования

def open_and_read_pdf(file_path: str) -> list[dict]:
    # Ваш код чтения PDF

def advanced_text_formatter(text: str) -> str:
    # Дополнительный препроцессинг

def split_list(input_list: list, slice_size: int = config["rag"]["num_sentences_per_chunk"]) -> list[list]:
    # Функция сплита

def load_and_chunk_data():
    texts = []
    for book in config["data"]["books"]:
        texts.extend(open_and_read_pdf(config["data"]["pdf_path"] + book + '.pdf'))
    
    # Далее: препроцессинг, чанкинг как в ноутбуке
    # Возвращайте pages_and_chunks
    return pages_and_chunks