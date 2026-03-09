import os
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

import os
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

print("Setting up the AI engines...")

# 1. Set the Global Settings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. Hardcode the key directly into the Groq engine to override any broken .env files.
# REPLACE the string below with a BRAND NEW key from the Groq console!
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=groq_key
) 

# 3. Load the pre-built index from your storage folder
STORAGE_DIR = "./storage"
print(f"Loading database from '{STORAGE_DIR}'...")

try:
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
except Exception as e:
    print(f"Error loading the index: {e}")
    print("Make sure you ran ingest_local.py first and the 'storage' folder exists.")
    exit()

# 4. Create the Chat Engine
query_engine = index.as_query_engine(streaming=True)
print("\n✅ Chatbot is ready! Type 'exit' to quit.\n")

# 5. Interactive Chat Loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Goodbye!")
        break
    
    print("Bot: ", end="")
    try:
        # We use stream_chat for a cool typewriter effect since Groq is so fast!
        response = query_engine.query(user_input)
        for text in response.response_gen:
            print(text, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\n[An error occurred: {e}]\n")