# src/app.py
import os
import asyncio
import logging
from functools import lru_cache
from typing import List
import PyPDF2

# LlamaIndex imports for Chroma vector store
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq

# Logging setup
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported file types
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx'}

def is_supported_file(filepath: str) -> bool:
    return os.path.splitext(filepath)[1].lower() in SUPPORTED_EXTENSIONS

def scan_data_directory(base_path: str) -> List[str]:
    file_paths = []
    try:
        logger.info(f"Scanning directory: {base_path}")
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                if is_supported_file(file_path):
                    file_paths.append(file_path)
                    logger.debug(f"Found supported file: {file_path}")
    except Exception as e:
        logger.error(f"Error scanning directory {base_path}: {e}")
    return file_paths

@lru_cache(maxsize=1000)
def load_document_sync(file_path: str):
    try:
        if file_path.lower().endswith('.pdf'):
            content = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
                    if len(content) > 100_000:
                        content = content[:100_000] + "\n[Content truncated]"
                        break
            return content.strip() if content.strip() else None
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(100_000)  # Limit size
            return content.strip()
        else:
            logger.warning(f"No loader for {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

async def add_documents_to_index_in_batches(documents, storage_context: StorageContext):
    MAX_DOCUMENTS_PER_BATCH = 50
    index = None
    for i in range(0, len(documents), MAX_DOCUMENTS_PER_BATCH):
        batch = documents[i:i + MAX_DOCUMENTS_PER_BATCH]
        if index is None:
            index = VectorStoreIndex.from_documents(
                batch,
                storage_context=storage_context,
                embed_model=LlamaSettings.embed_model
            )
        else:
            for doc in batch:
                index.insert(doc)
        await asyncio.sleep(1)
    return index
