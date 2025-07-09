import os, sys
import pandas as pd
from chatbot.nodes_constructor import (
    MasterNode,
)
from chatbot.prompts import Prompts
from langgraph.prebuilt import ToolNode
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL=os.getenv("MODEL")

current_path = os.path.dirname(os.path.abspath(__file__))

MASTER_NODE_PROMPT = Prompts.master.get_prompt()

master_node=MasterNode(llm=MODEL, instructions=MASTER_NODE_PROMPT)