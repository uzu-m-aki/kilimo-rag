from llama_index.llms.groq import Groq
from llama_index.core import Settings
from config import GROQ_API_KEY, MODEL_NAME

Settings.llm = Groq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY
)
