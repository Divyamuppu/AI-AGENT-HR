import os
import streamlit as st
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

st.markdown("""
<style>
html, body {
    height: 100%;
    background: linear-gradient(
        135deg,
        #e0ecff,
        #c7ddff,
        #b6d4ff
    ) !important;
}

/* Streamlit main container */
.stApp {
    background: linear-gradient(
        135deg,
        #e0ecff,
        #c7ddff,
        #b6d4ff
    ) !important;
}

/* Streamlit internal wrapper */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #e0ecff,
        #c7ddff,
        #b6d4ff
    ) !important;
}
</style>
""", unsafe_allow_html=True)


# ===========================
# STREAMLIT CONFIG
# ===========================
st.set_page_config(page_title="HR AI Agent", layout="centered")

# ===========================
# GLOBAL BACKGROUND (BLUE)
# ===========================
# st.markdown("""
# <style>
# html, body, [class*="css"] {
#     background: linear-gradient(
#         135deg,
#         #e0ecff,
#         #c7ddff,
#         #b6d4ff
#     ) !important;
# }
# </style>
# """, unsafe_allow_html=True)

# ===========================
# UI STYLING
# ===========================
st.markdown("""
<style>
.block-container {
    max-width: 720px;
    margin-top: 12vh;
    background: white;
    border-radius: 20px;
    padding: 2.5rem;

    /* NEW */
    border: 2px solid #000000;
    box-shadow: 0 18px 40px rgba(0,0,0,0.25);
}

h1 {
    text-align: center;
    color: #1f2937;
}

input {
    background: #f9fafb !important;
    border-radius: 10px !important;
    border: 1px solid #000000 !important;
}

.stButton>button {
    background: #2563eb !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600;
    padding: 0.65rem 1.4rem;
    width: 100%;
}

.stButton>button:hover {
    background: #1e40af !important;
}

.card {
    background: #f1f5f9;
    border: 1px solid #000000;
    border-radius: 14px;
    padding: 1rem;
    cursor: pointer;
    text-align: center;
    font-size: 0.95rem;
    color: #1f2937;
    transition: all 0.2s ease;
}

.card:hover {
    background: #e0e7ff;
    transform: translateY(-3px);
}

.stSuccess {
    background: #eff6ff !important;
    border-left: 6px solid #2563eb;
    padding: 16px;
    border-radius: 12px;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# GEMINI
# ===========================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = genai.GenerativeModel("models/gemini-2.5-flash")

# ===========================
# HELPERS
# ===========================
def load_docs():
    return [
        open(f"data/{f}", encoding="utf-8").read()
        for f in os.listdir("data")
    ]

@st.cache_resource
def embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def chroma_client():
    return chromadb.Client(
        settings=chromadb.Settings(persist_directory="chroma_db")
    )

# ===========================
# INIT DATABASES
# ===========================
embed = embedder()
client = chroma_client()

policy_db = client.get_or_create_collection("hr_policies")
memory_db = client.get_or_create_collection("qa_memory")

if policy_db.count() == 0:
    for i, doc in enumerate(load_docs()):
        policy_db.add(
            documents=[doc],
            ids=[str(i)],
            embeddings=[embed.encode(doc).tolist()]
        )

# ===========================
# LLM HELPERS
# ===========================
def summarize_from_policy(context, question):
    prompt = f"""
Answer in ONE short sentence.
Use ONLY the policy text.

Policy:
{context}

Question:
{question}
"""
    return llm.generate_content(prompt).text.strip()

def llm_fallback(question):
    prompt = f"""
Answer in ONE short sentence.
If unsure, say: "Not specified in current HR policy."

Question:
{question}
"""
    return llm.generate_content(prompt).text.strip()

# ===========================
# UI
# ===========================
st.title("🤖 HR AI Agent")

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

question = st.text_input(
    "Ask your HR question",
    placeholder="e.g., What is the notice period?",
    value=st.session_state.question_input
)

# -------- GET ANSWER BUTTON --------
if st.button("Get Answer"):
    if question.strip():
        q_embedding = embed.encode(question).tolist()

        # STEP 1: MEMORY
        mem = memory_db.query(query_embeddings=[q_embedding], n_results=1)

        if (
            mem.get("documents")
            and mem["documents"][0]
            and mem.get("distances")
            and mem["distances"][0][0] < 0.15
        ):
            st.info("🔁 Answered from memory")
            answer = mem["documents"][0][0]

        else:
            # STEP 2: POLICY
            pol = policy_db.query(query_embeddings=[q_embedding], n_results=1)
            policy_context = pol["documents"][0][0] if pol.get("documents") else ""

            if policy_context.strip():
                answer = summarize_from_policy(policy_context, question)
            else:
                answer = llm_fallback(question)

            memory_db.add(
                documents=[answer],
                embeddings=[q_embedding],
                metadatas=[{"question": question}],
                ids=[f"mem_{memory_db.count()}"]
            )

        st.success(answer)
    else:
        st.warning("Please enter a question.")

# -------- FREQUENTLY ASKED --------
st.markdown("### 💡 Frequently Asked")

cols = st.columns(3)
suggestions = [
    "What is the notice period?",
    "Can I take casual leave?",
    "What is the WFH policy?"
]

for col, text in zip(cols, suggestions):
    with col:
        if st.button(text, key=text):
            st.session_state.question_input = text
            st.rerun()
