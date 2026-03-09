import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("🌱 Waking up the Kilimo Ingestion Engine...")

# --- 1. Setup the AI Brain (Secure Mode) ---
# We intentionally DO NOT include Groq or your API key here.
# HuggingFaceEmbedding runs locally on your machine to do the math.
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- 2. The Magic Scissors (Chunking) ---
# Cuts PDFs into safe, 512-word paragraphs so the AI never crashes.
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

DATA_DIR = "./data"
STORAGE_DIR = "./storage"

# --- 3. Read and Process ---
print(f"📚 Scanning folder '{DATA_DIR}' for agricultural documents...")
try:
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"✅ Found and loaded {len(documents)} document pages.")
except Exception as e:
    print(f"❌ Error reading documents: {e}")
    exit()

# --- 4. Embed and Index ---
print("🧠 Chopping text into chunks and calculating mathematical vectors. Please wait...")
index = VectorStoreIndex.from_documents(documents)

# --- 5. Save to Database ---
print(f"💾 Saving clean, safe knowledge to '{STORAGE_DIR}'...")
index.storage_context.persist(persist_dir=STORAGE_DIR)

print("🎉 Success! Your database is clean, key-free, and safe for GitHub.")