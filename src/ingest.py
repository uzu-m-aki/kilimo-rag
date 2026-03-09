from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os

DATA_PATH = "../Data"

def create_index():
    documents = SimpleDirectoryReader(DATA_PATH).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

if __name__ == "__main__":
    index = create_index()
    print("Index created successfully!")
