import os, sys
import pandas as pd
from chatbot.nodes_constructor import (
    MasterNode,
    PredictorNode,
    RagNode,
    LiteratureNode
)
from chatbot.prompts import Prompts
from chatbot.literature_functions import LiteratureTools
from langgraph.prebuilt import ToolNode
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL=os.getenv("MODEL")

current_path = os.path.dirname(os.path.abspath(__file__))

MASTER_NODE_PROMPT = Prompts.master.get_prompt()
PREDICTOR_NODE_PROMPT = Prompts.predict.get_prompt()
RAG_NODE_PROMPT = Prompts.rag.get_prompt()
LITERATURE_NODE_PROMPT = Prompts.literature.get_prompt()

master_node=MasterNode(llm=MODEL, instructions=MASTER_NODE_PROMPT)
predictor_node=PredictorNode(llm=MODEL, instructions=PREDICTOR_NODE_PROMPT)
rag_node=RagNode(llm=MODEL, instructions=RAG_NODE_PROMPT)
literature_node = LiteratureNode(llm=MODEL, instructions=LITERATURE_NODE_PROMPT, functions=LiteratureTools)