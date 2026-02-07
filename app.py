import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai


# ===========================
# LOAD ENV
# ===========================
load_dotenv()

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(page_title="HR AI Agent", layout="centered")

# ===========================
# STYLES
# ===========================
st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(135deg,#e0ecff,#c7ddff,#b6d4ff)!important;
}
.block-container {
    max-width:760px;
    margin-top:15vh;
    background:white;
    border-radius:22px;
    padding:2.5rem;
    border:2px solid black;
}
/* Question input box ONLY – black border */
div[data-testid="stTextInput"] input {
    border: 2px solid #000 !important;
    border-radius: 12px !important;
    padding: 10px !important;
}


.stButton > button {
    border-radius:16px;
    font-weight:600;
    border:2px solid #000;
    background:#1d4ed8;
    color:white;
    height:64px;
}
.app-title {
    text-align:center;
    font-size:3rem;
    font-weight:700;
    margin-bottom:1.5rem;
}
/* FAQ / Pluck buttons – FINAL, WORKING */
div[data-testid="stColumn"] .stButton > button {
    border-radius:16px !important;
    font-weight:600 !important;
    border:2px solid #000 !important;
    background:linear-gradient(135deg,#1e3a8a,#1d4ed8) !important;
    color:#ffffff !important;
    height:64px;
    transition: all 0.25s ease;
}

div[data-testid="stColumn"] .stButton > button:hover {
    transform: translateY(-3px);
    box-shadow:0 10px 20px rgba(0,0,0,0.35);
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='app-title'>🤖 HR AI Agent</div>", unsafe_allow_html=True)

# ===========================
# GEMINI CONFIG
# ===========================
from google import genai
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents=prompt
)


# ===========================
# EMBEDDINGS
# ===========================
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()

# ===========================
# CHROMADB (PERSISTENT)
# ===========================
client = chromadb.Client(
    settings=chromadb.Settings(persist_directory="chroma_db")
)

policy_db = client.get_or_create_collection("hr_policies")
memory_db = client.get_or_create_collection("qa_memory")

# # 🔥 TEMPORARY – RUN ONCE
# memory_db.delete(where={})



def load_policies():
    docs = []
    for file in os.listdir("data"):
        if file.endswith(".txt"):
            with open(os.path.join("data", file), encoding="utf-8") as f:
                text = f.read()

                # Split by sentences, not lines
                for chunk in text.split("."):
                    chunk = chunk.strip()
                    if len(chunk) > 30:
                        docs.append(chunk + ".")
    return docs



if policy_db.count() == 0:
    for i, doc in enumerate(load_policies()):
        policy_db.add(
            ids=[f"policy_{i}"],
            documents=[doc],
            embeddings=[embedder.encode(doc).tolist()]
        )

# ===========================
# UI (TYPE OR CLICK – BOTH WORK)
# ===========================
if "q" not in st.session_state:
    st.session_state.q = ""

question = st.text_input(
    "Ask your HR question",
    value=st.session_state.q,
    placeholder="Can leaves be carried forward?"
)

st.markdown("##### 💡 Frequently Asked Questions")

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Can leaves be carried forward?", use_container_width=True):
        st.session_state.q = "Can leaves be carried forward?"
        st.rerun()

with c2:
    if st.button("What is the work from home policy?", use_container_width=True):
        st.session_state.q = "What is the work from home policy?"
        st.rerun()

with c3:
    if st.button("What is the notice period?", use_container_width=True):
        st.session_state.q = "What is the notice period?"
        st.rerun()



# ===========================
# ANSWER FLOW (STRICT ORDER)
# ===========================
if question.strip():
    q_emb = embedder.encode(question).tolist()

    # 1️⃣ QA MEMORY (HIGHEST PRIORITY)
    mem = memory_db.query(
        query_embeddings=[q_emb],
        n_results=1,
        include=["documents", "distances"]
    )

    if mem["documents"] and mem["documents"][0] and mem["distances"][0][0] < 0.15:
        st.info("Answered from QA memory")
        st.success(mem["documents"][0][0])
        st.stop()

    # 2️⃣ POLICY DOCUMENTS (RAG – MUST CHECK)
    pol = policy_db.query(
        query_embeddings=[q_emb],
        n_results=1,
        include=["documents"]
    )

    if pol["documents"] and pol["documents"][0]:
        policy_text = pol["documents"][0][0]

        prompt = f"""
Extract the answer ONLY if it is explicitly stated in the policy text.
If the answer is not explicitly stated, return exactly: NOT_FOUND

Policy:
{policy_text}

Question:
{question}

Return ONE short factual sentence.
"""
        answer = ask_gemini(prompt)

        if answer.strip().upper() != "NOT_FOUND":
            memory_db.add(
                ids=[f"mem_{memory_db.count()}"],
                documents=[answer],
                embeddings=[q_emb],
                metadatas=[{"question": question, "source": "policy"}]
            )
            st.success(answer)
            st.stop()

    # 3️⃣ LLM GENERATION (LAST RESORT ONLY)
    prompt = f"""
You are an internal HR assistant for this company.
Answer briefly in ONE sentence.

Question:
{question}
"""
    answer = ask_gemini(prompt)

    memory_db.add(
        ids=[f"mem_{memory_db.count()}"],
        documents=[answer],
        embeddings=[q_emb],
        metadatas=[{"question": question, "source": "llm"}]
    )

    st.success(answer)
