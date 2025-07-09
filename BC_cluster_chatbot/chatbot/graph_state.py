from typing import Any, Dict, List, Optional, TypedDict
from pandas import DataFrame
# Environment variables
from dotenv import load_dotenv
# LangChain and Google AI specific libraries
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage
)


class GraphState(TypedDict):
    session_id: str
    messages: List[BaseMessage]
    expression_data: Optional[DataFrame]
    clinical_data: Optional[Dict[str, Any]]
    patient_data: str
    answer: str
    response: Optional[AIMessage]
    original_query: Optional[HumanMessage]
    answer_source: Optional[str]
    history: List[BaseMessage] 