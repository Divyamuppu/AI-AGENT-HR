import os
import streamlit as st
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer

# ===========================
# STREAMLIT CONFIG
# ===========================
st.set_page_config(page_title="HR AI Agent (Agentic)")

# ===========================
# UI STYLING
# ===========================
st.markdown("""
<style>
.stApp { background-color: #f8fafc; }
.block-container {
    max-width: 900px;
    margin-top: 3rem;
    background: rgba(255,255,255,0.9);
    border-radius: 18px;
    padding: 2.5rem;
}
h1 { text-align: center; }
input { background: white !important; border-radius: 8px !important; }
.stSuccess {
    background: #eff6ff !important;
    border-left: 6px solid #3b82f6;
    padding: 16px;
    border-radius: 10px;
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

# =====================================
#  CONVERTING THE POLICY DOCUMENET INTO EMBEDDINGS
# ==========================================

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
Do NOT generalize.

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

question = st.text_input("Ask your HR question:")

if question:
    q_embedding = embed.encode(question).tolist()

    # ==================================================
    # STEP 1: MEMORY FIRST (SELF-RECURSIVE)
    # ==================================================
    mem = memory_db.query(query_embeddings=[q_embedding], n_results=1)

    if (
        mem.get("documents")
        and mem["documents"][0]
        and mem.get("distances")
        and mem["distances"][0][0] < 0.15
    ):
        st.info("🔁 Answered from memory")
        st.success(mem["documents"][0][0])

    else:
        # ==================================================
        # STEP 2: POLICY DOCUMENTS (MANDATORY)
        # ==================================================
        pol = policy_db.query(query_embeddings=[q_embedding], n_results=1)

        policy_context = ""
        if pol.get("documents") and pol["documents"][0]:
            policy_context = pol["documents"][0][0]

        # ==================================================
        # STEP 3: DECISION
        # ==================================================
        if policy_context.strip():
            # ✅ ALWAYS answer from policy if policy exists
            answer = summarize_from_policy(policy_context, question)
        else:
            # 🔥 Only if policy truly does not exist
            answer = llm_fallback(question)

        st.success(answer)

        # ==================================================
        # STEP 4: SAVE TO MEMORY
        # ==================================================
        memory_db.add(
            documents=[answer],
            embeddings=[q_embedding],
            ids=[f"mem_{memory_db.count()}"]
        )
