import streamlit as st
import os
import io
import librosa
import numpy as np
import torch
from transformers import pipeline
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# --- 1. Page Configuration (Must be first) ---
st.set_page_config(page_title="Kilimo AI", page_icon="✨", layout="centered")

# --- 2. Premium Dark Mode CSS Injection ---
st.markdown("""
<style>
    /* Deep dark background like Gemini/ChatGPT */
    .stApp {
        background-color: #131314;
        color: #e3e3e3;
    }
    
    /* Sleek, slightly lighter sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e1f22;
        border-right: 1px solid #33363a;
    }
    
    /* Gradient glowing title */
    .kilimo-title {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #81c784, #4caf50, #2e7d32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-top: 20px;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Subtle subtitle */
    .kilimo-subtitle {
        text-align: center;
        color: #9aa0a6;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }

    /* Modern, hollow buttons with hover glow */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #4CAF50;
        background-color: transparent;
        color: #81c784;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: rgba(76, 175, 80, 0.1);
        color: #a5d6a7;
        border-color: #a5d6a7;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.2);
    }

    /* Style the chat input box to look embedded */
    [data-testid="stChatInput"] {
        background-color: #1e1f22 !important;
        border: 1px solid #33363a !important;
        border-radius: 24px !important;
    }

    /* Subtly separate user messages from AI messages */
    [data-testid="stChatMessage"] {
        background-color: transparent;
        border-radius: 12px;
        padding: 1.5rem;
    }
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: rgba(30, 31, 34, 0.5); /* Faint background for user */
    }
</style>
""", unsafe_allow_html=True)

from dotenv import load_dotenv
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

# --- 3. Sidebar & Settings ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #e3e3e3;'>⚙️ Mipangilio</h2>", unsafe_allow_html=True)
    
    lang_choice = st.selectbox("🗣️ Lugha / Language", ["Swahili", "Kikuyu", "English"])
    
    # ⚠️ PASTE YOUR GROQ KEY HERE ⚠️
    st.secrets.groq_key = groq_key
    
    st.divider()
    st.caption("Kumbukumbu (Memory Context)")
    if st.button("✨ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_voice_query = None
        st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
        st.rerun()

# --- Custom Header ---
st.markdown('<p class="kilimo-title">Kilimo Expert AI</p>', unsafe_allow_html=True)
st.markdown(f'<p class="kilimo-subtitle">Your personal agronomist, currently speaking <b>{lang_choice}</b></p>', unsafe_allow_html=True)

# --- 4. Explicit LLM & Embedding Initialization ---
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Groq(model="llama-3.1-8b-instant", api_key=groq_key, temperature=0.0)

Settings.embed_model = embed_model
Settings.llm = llm

# --- 5. Load Heavy Models (Cached) ---
@st.cache_resource(show_spinner="Waking up the neural agronomist... ✨")
def load_heavy_engines():
    k_asr = pipeline("automatic-speech-recognition", model="badrex/w2v-bert-2.0-kikuyu-asr", device=-1)
    s_asr = pipeline("automatic-speech-recognition", model="thinkKenya/wav2vec2-large-xls-r-300m-sw", device=-1)
    
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    return k_asr, s_asr, index

k_asr, s_asr, index = load_heavy_engines()

# --- 6. Session State Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_voice_query" not in st.session_state:
    st.session_state.pending_voice_query = None
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=1000)

# --- 7. Build Chat Engine ---
system_prompt = (
    f"You are a friendly, professional Senior Agronomist in Kenya. "
    f"You MUST always respond fluently in {lang_choice}. "
    f"You are happy to engage in normal conversation, small talk, and greetings."
)

context_template = (
    "Agricultural Context from your records is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "CRITICAL RULES:\n"
    "1. SMALL TALK: If the user is greeting you (e.g., 'habari', 'sasa', 'hello', 'mambo'), IGNORE the context completely. Respond naturally and politely.\n"
    "2. FARMING QUESTIONS: Use ONLY the context records above to answer.\n"
    f"3. NO GUESSING: If not in records, say 'I do not have that specific information in my records' in {lang_choice}.\n"
    "4. NO METADATA: Never output 'page_label' or 'file_path'."
)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=st.session_state.chat_memory,
    system_prompt=system_prompt,
    context_prompt=context_template, 
    similarity_top_k=2, 
    llm=llm 
)

# --- 8. Display Chat History with Modern Avatars ---
for message in st.session_state.messages:
    # 🧑‍🌾 for user, ✨ for the AI (cleaner than the robot)
    avatar_icon = "🧑‍🌾" if message["role"] == "user" else "✨"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# --- 9. Inputs (Voice & Text) ---
input_container = st.container()

with input_container:
    audio_input = st.audio_input(f"🎙️ Speak in {lang_choice}")
    typed_query = st.chat_input("Message Kilimo AI...")

if audio_input:
    audio_id = hash(audio_input.getvalue())
    if st.session_state.get("last_audio_id") != audio_id:
        with st.spinner("Processing audio..."):
            try:
                audio_bytes = audio_input.read()
                y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                y_float32 = np.asarray(y, dtype=np.float32)
                
                if lang_choice == "Kikuyu":
                    transcribed = k_asr(y_float32)["text"]
                elif lang_choice == "Swahili":
                    transcribed = s_asr(y_float32)["text"]
                else:
                    transcribed = "Voice input is currently optimized for Swahili and Kikuyu."
                
                st.session_state.last_audio_id = audio_id
                
                if transcribed.strip():
                    st.session_state.pending_voice_query = transcribed
                    st.rerun() 
                else:
                    st.warning("⚠️ Audio unclear. Please try again.")
            except Exception as e:
                st.error(f"Audio Error: {e}")

# --- 10. The Response Loop ---
final_query = st.session_state.pending_voice_query or typed_query

if final_query:
    st.session_state.pending_voice_query = None 
    
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(final_query)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Synthesizing records..."):
            response = chat_engine.chat(final_query)
            st.markdown(response.response)
    
    st.session_state.messages.append({"role": "assistant", "content": response.response})