from ingest import create_index
import llm  # Ensures Groq is loaded

def main():
    index = create_index()
    query_engine = index.as_query_engine()

    while True:
        question = input("Ask a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        response = query_engine.query(question)
        print("\nAnswer:", response)

if __name__ == "__main__":
    main()
