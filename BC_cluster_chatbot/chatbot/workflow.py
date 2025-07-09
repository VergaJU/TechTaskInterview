from typing import List, Literal
import logging
import chatbot.nodes as nodes
from chatbot.graph_state import GraphState
from langgraph.graph import END, StateGraph
from google.genai.types import GenerateContentResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bc_chatbot.log'),
        logging.StreamHandler()
    ]
)


