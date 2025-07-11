import os
import streamlit as st
import pandas as pd
import uuid
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from chatbot.nodes import master_node 
from chatbot.workflow import ChatWorkflow
#### create or load vector database:
from PIL import Image

from typing import TypedDict # For GraphState type hinting

# Import the database manager class
from chatbot.create_db import HtmlVectorDatabaseManager, DEFAULT_CORPUS_DIR, DEFAULT_CHROMA_DB_DIR, DEFAULT_EMBEDDINGS_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


CORPUS_DIR = os.getenv("CORPUS_DIR", DEFAULT_CORPUS_DIR)
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", DEFAULT_CHROMA_DB_DIR)
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", DEFAULT_EMBEDDINGS_MODEL)
LLM_MODEL_NAME = os.getenv("MODEL", "gemini-2.0-flash")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
K_RETRIEVAL = int(os.getenv("K_RETRIEVAL", 5)) 


if "messages" not in st.session_state:
    st.session_state.messages = []


if 'db_manager' not in st.session_state:
    print("Initializing HtmlVectorDatabaseManager...")
    st.session_state['db_manager'] = HtmlVectorDatabaseManager(
        corpus_dir=CORPUS_DIR,
        chroma_db_dir=CHROMA_DB_DIR,
        embeddings_model_name=EMBEDDINGS_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    print("HtmlVectorDatabaseManager stored in session_state.")


if 'vector_store' not in st.session_state or st.session_state['vector_store'] is None:
    print("Initializing or loading vector store...")
    # initialize_vector_store returns the Chroma instance or None on failure
    st.session_state['vector_store'] = st.session_state['db_manager'].initialize_vector_store(force_reindex=False)

    if st.session_state['vector_store'] is None:
        st.error("Failed to initialize or load vector database. RAG functionality will be unavailable.")
        # You might want to stop the app or disable related features here
    else:
        print("Vector store initialized/loaded and stored in session_state.")



st.set_page_config(
    page_title="Breast Cancer Cluster Chatbot",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)



# Only assign a UUID if one doesn't already exist in this session
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())


with st.sidebar:
    st.markdown("""## Patient Data Input
    Load a .csv file with two columns: `gene_id` and `expression`""")

    uploaded_file = st.file_uploader("Patient Expression Profile", key="patient_expression", type="csv")

    if st.button("Load Example File", key="load_example_button"):
        example_path = "/app/chatbot/Data/test_expression.csv"
        if os.path.exists(example_path):
            st.session_state.patient_expression_df = pd.read_csv(example_path)
            st.success("Example file loaded!")
            uploaded_file = True
        else:
            st.error("Example file not found.")
            uploaded_file = None

    age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
    tumor_size = st.number_input("Tumor Size", min_value=0, max_value=150, value=15, step=1)
    lymph_node = st.selectbox("Lymph Node Status", ["NA", "Positive", "Negative"], index=0)
    er_status = st.selectbox("ER Status", ["NA", "Positive", "Negative"], index=0)
    pgr_status = st.selectbox("PR Status", ["NA", "Positive", "Negative"], index=0)
    her2_status = st.selectbox("HER2 Status", ["NA", "Positive", "Negative"], index=0)
    ki67_status = st.selectbox("Ki67 Status", ["NA", "Positive", "Negative"], index=0)
    nhg = st.selectbox("Nuclear Grade", ["NA", "G1", "G2", "G3"], index=0)
    pam50 = st.selectbox("PAM50", ['NA', 'LumA', 'LumB', 'HER2', 'Basal', 'Normal'], index=0)


svg_path = "/app/chatbot/asset/logo.svg"
with open(svg_path, "r") as f:
    svg = f.read()
    
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 100px;">
        <div style="width:75px;">{svg}</div>
        <h1 style="margin: 0;">Breast Cancer Cluster Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown("""
Welcome to the **Breast Cancer Cluster Chatbot** — an interactive assistant designed to help you explore and interpret **patient-specific clustering results** based on gene expression and clinical data.

**How it works:**

- Upload a **gene expression profile** for a patient.
- Provide **clinical details** (e.g., age, tumor characteristics) using the sidebar.
- Ask questions about the **patient’s cluster**, associated **biological pathways**, or broader **breast cancer knowledge**.

The chatbot uses:
- A **predictive model** to classify the patient and interpret results (including SHAP and GSEA).
- A **retrieval system** to answer questions about clusters, PAM50 subtypes, NHG grades, and survival data.
- A **literature node** for general breast cancer insights from scientific sources.

Each user session is isolated and secure. All figures and results are processed and shown only for your session.
""")


# Only parse if file is present
if uploaded_file is not None and not isinstance(uploaded_file, bool):
    try:
        st.session_state.patient_expression_df = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded and loaded!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        if 'patient_expression_df' in st.session_state:
            del st.session_state.patient_expression_df

clinical_data = {
    "age": age,
    "tumor_size": tumor_size,
    "lymph_node": lymph_node,
    "er_status": er_status,
    "pgr_status": pgr_status,
    "her2_status": her2_status,
    "ki67_status": ki67_status,
    "nhg": nhg,
    "pam50": pam50
}

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"], unsafe_allow_html=True)


question = st.chat_input(
    "Ask information about patients cluster analysis",
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)
    state = {
        "session_id": st.session_state.user_id,
        "expression_data": st.session_state.get('patient_expression_df', None),
        "clinical_data": clinical_data,
        "messages": question,
        "vector_db": st.session_state.get('vector_store', None),
        "k_retrieval": K_RETRIEVAL,
    }
    workflow=ChatWorkflow()
    result=workflow.run(state)
    assistant_response = result.get("response", "")
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    st.chat_message("assistant").markdown(assistant_response, unsafe_allow_html=True)
