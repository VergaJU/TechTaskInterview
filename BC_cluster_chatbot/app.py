import os
import streamlit as st
import pandas as pd
import uuid
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from chatbot.nodes import master_node 

st.set_page_config(
    page_title="BC cluster chatbot",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)


# Only assign a UUID if one doesn't already exist in this session
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())


with st.sidebar:
    patient_expression = st.file_uploader("Patient Expression Profile: gene_id, expression", key="patient_expression", type="csv")
    age =st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
    tumor_size = st.number_input("Tumor Size", min_value=0, max_value=150, value=15, step=1)
    lymph_node = st.selectbox("Lymph Node Status", ["NA", "Positive", "Negative"], index=0)
    er_status = st.selectbox("ER Status", ["NA", "Positive", "Negative"], index=0)
    pgr_status = st.selectbox("PR Status", ["NA", "Positive", "Negative"], index=0)
    her2_status = st.selectbox("HER2 Status", ["NA", "Positive", "Negative"], index=0)
    ki67_status = st.selectbox("Ki67 Status", ["NA", "Positive", "Negative"], index=0)
    nhg = st.selectbox("Nuclear Grade", ["NA", "G1", "G2", "G3"], index=0)
    pam50 = st.selectbox("PAM50", ['NA', 'LumA', 'LumB', 'HER2', 'Basal', 'Normal'], index=0)

st.title("BC Cluster Chatbot")
st.write("This is a chatbot for breast cancer cluster analysis. Please upload the patient expression profile and fill in the details in the sidebar.")

question = st.text_input(
    "Ask information about patients cluster analysis",
    placeholder="Which cluster the patient belongs to?",
    disabled=not patient_expression,
)


if patient_expression:
    patient_expression = pd.read_csv(patient_expression)[['gene_id', 'expression']]


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



if question:
    response=master_node.get_node(question)
    st.write(response)