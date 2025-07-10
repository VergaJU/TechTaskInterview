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



class ChatWorkflow:
    def __init__(self):
        self.graph = self._build()

    def start_nodes(self) -> List[str]:
        """
        Initialize nodes and node names
        """
        self.master_node = nodes.master_node
        self.predictor_node = nodes.predictor_node
        self.rag_node = nodes.rag_node
        self.literature_node = nodes.literature_node
        logging.info("Starting nodes: master and predictor.")
        # Return the names of the starting nodes
        self.MASTER_NODE="master"
        self.PREDICTOR_NODE="predictor"
        self.RAG_NODE="rag"
        self.LITERATURE_NODE="literature"

    def route_master(self, state: GraphState) -> Literal["predictor"]:
        """
        Route the state to the predictor node based on the master's response.
        """
        logging.info("Routing from master node.")
        response=state["response"]
        if "***ROUTE_TO_PREDICTOR***" in response:
            logging.info("Routing to predictor node.")
            return self.PREDICTOR_NODE
        elif "***ROUTE_TO_RAG***" in response:
            logging.info("Routing to RAG node.")
            return self.RAG_NODE
        elif "***ROUTE_TO_LITERATURE***" in response:
            logging.info("Routing to Literature node.")
            return self.LITERATURE_NODE
        else:
            logging.info("No routing to predictor node, ending workflow.")
            return END

    def _build(self):
        self.start_nodes()
        builder = StateGraph(GraphState)
        builder.add_node(self.MASTER_NODE, self.master_node.get_node)
        builder.add_node(self.PREDICTOR_NODE, self.predictor_node.get_node)
        builder.add_node(self.RAG_NODE, self.rag_node.get_node)
        builder.add_node(self.LITERATURE_NODE, self.literature_node.get_node)

        builder.set_entry_point(self.MASTER_NODE)
        builder.add_conditional_edges(self.MASTER_NODE,
                                      self.route_master,
                                      {
                                          self.PREDICTOR_NODE: self.PREDICTOR_NODE,
                                          self.RAG_NODE: self.RAG_NODE,
                                          self.LITERATURE_NODE: self.LITERATURE_NODE,
                                          END: END
                                      })

        builder.add_edge(self.PREDICTOR_NODE, END)
        builder.add_edge(self.RAG_NODE, END)
        builder.add_edge(self.LITERATURE_NODE, END)
        return builder.compile()

    def run(self, state):
        # optionally modify or validate state here
        return self.graph.invoke(state)