"""
src/rag_graph.py — LangGraph RAG: retrieve → generate (LLM version)
Простой, чистый, готов к FastAPI
"""

from typing import TypedDict, Annotated, Sequence
from operator import add
import logging

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.config import get_config
from src.vector_store import get_vector_store_manager
from src.llm import get_llm

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    question: str
    context: Annotated[Sequence[Document], add]
    response: str
    messages: Annotated[Sequence[AIMessage | HumanMessage], add]

def retrieve(state: GraphState) -> GraphState:
    """🔍 Поиск релевантных чанков в Qdrant"""
    vs = get_vector_store_manager()
    docs = vs.search(state["question"], k=6)
    logger.debug(f"Found {len(docs)} chunks for '{state['question'][:50]}...'")
    return {"context": docs}

def generate(state: GraphState) -> GraphState:
    """🤖 Groq LLM генерация"""
    config = get_config()
    context = "\n\n".join([f"[{i+1}] {doc.page_content}" 
                          for i, doc in enumerate(state["context"])])
    
    prompt_text = f"""{config.prompt.system_role}

КОНТЕКСТ ИЗ КНИГ:
{context}

ВОПРОС: {state["question"]}

ОТВЕТ:"""
    
    llm = get_llm()  # llama-3.1-8b по умолчанию
    
    try:
        response = llm.invoke(prompt_text)
        return {
            "response": response.content,
            "messages": [HumanMessage(content=state["question"]), 
                        AIMessage(content=response.content)]
        }
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return {"response": f"Ошибка LLM: {str(e)}"}


# 🏗️ Граф
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_app = workflow.compile()

def ask(question: str, stream=False) -> str:
    """🎯 Главная RAG функция"""
    result = rag_app.invoke({"question": question})
    return result["response"]
