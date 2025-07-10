from typing import Optional, Dict, Any

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable



from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai import types
from google import genai 
from dotenv import load_dotenv
import os
import papermill as pm
import subprocess
import base64
import re
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bc_chatbot.log'),
        logging.StreamHandler()
    ]
)
load_dotenv()

current_path = os.path.dirname(os.path.abspath(__file__))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL=os.getenv("MODEL")

class node:
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None, logging_key = None):
        self.llm = llm
        self.instructions = instructions
        self.functions = functions
        self.welcome = welcome
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.logging_key = logging_key
    def get_node(self, state):
        return None
    def log_message(self, message):
        """Log the message to the console."""
        logging.info(self.logging_key + message)
        print(message)


class MasterNode(node):
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None):
        super().__init__(llm=llm, instructions=instructions, functions=functions, welcome=welcome, logging_key="Master node.- ")
        self.llm_master=ChatGoogleGenerativeAI(model=self.llm,
                                               temperature=0.5,)


    def run_model(self, messages):
        """Run the model with the given messages."""
        #print(f"--- Message going to the llm_master: {messages}---")
        response = self.llm_master.invoke(messages)
        return response
    
    def get_node(self, state):
        """The master node, which is the first node in the graph. It's role is to route the workflow to specific nodes and interface the user"""
        messages = state['messages']
        messages = [
            (
                "system",
                str(self.instructions)
            ),
            (
                "human",
                str(messages)
            )
        ]
        answer = None
        response=self.run_model(messages)
        state = state | {
            'response': response.content,
            'answer': answer
        }
        self.log_message("Master node response: " + state['response'])
        return state


class PredictorNode(node):
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None):
        super().__init__(llm=llm, instructions=instructions, functions=functions, welcome=welcome, logging_key="Predictor node.- ")
        self.llm_predictor = ChatGoogleGenerativeAI(model=self.llm)

    def compile_notebook(self, session_id, patient_expression, clinical_data):
        """Compile the notebook with the given session_id, patient_expression and clinical_data."""
        # make session_id directory
        session_dir = os.path.join('/app/sessions', session_id)
        os.makedirs(session_dir, exist_ok=True)
        # save patient_expression to csv
        if isinstance(patient_expression, pd.DataFrame):
            expression_path = os.path.join(session_dir, 'patient_expression.csv')
            patient_expression.to_csv(expression_path, index=False)
        notebook_path = '/app/notebooks/Sample_report.ipynb'
        output_path = os.path.join(session_dir, 'Sample_report_' + session_id + '.ipynb')
        logging.info(f"Compiling notebook: {notebook_path} to {output_path}")

        pm.execute_notebook(
            input_path=notebook_path,
            output_path=output_path,
            parameters={
                'sample': expression_path,
                'clinical_data': clinical_data
            }
        )
        return output_path
    
    def convert_notebook_to_md(self, notebook_path):
        """Convert the notebook to Markdown."""
        output_md = notebook_path.replace('.ipynb', '.md')
        subprocess.run([
            "jupyter", "nbconvert", "--to", "markdown", "--no-input",
            notebook_path, "--output", output_md
        ])
        return output_md

    
    def embed_images_in_markdown(self, markdown: str, patient_dir: str) -> str:
        """
        Replaces image paths in the markdown with base64-encoded inline images.
        Only works for PNGs in the given patient_dir.
        """
        def encode_image(match):
            image_path = match.group(1)  # captured path
            abs_path = Path(image_path)
            if not abs_path.is_absolute():
                abs_path = Path(patient_dir) / image_path
            try:
                with open(abs_path, "rb") as img:
                    encoded = base64.b64encode(img.read()).decode()
                return f"![png](data:image/png;base64,{encoded})"
            except Exception as e:
                return f"**[Image load failed: {e}]**"

        # Match ![alt](path) pattern
        return re.sub(r'!\[.*?\]\((.*?)\)', encode_image, markdown)

    def set_model(self, model):
        self.llm_predictor = ChatGoogleGenerativeAI(model=model)

    def run_model(self, messages):
        """Run the model with the given messages."""
        response = self.llm_predictor.invoke(messages)
        return response

    def get_node(self,state):
        """The predictor node, which is the second node in the graph. It's role is to predict the cluster of the patient and convert the notebook into an html report"""
        expression_data = state['expression_data']
        clinical_data = state['clinical_data']
        # Compile the notebook
        session_id = state['session_id']
        notebook_path = self.compile_notebook(session_id, expression_data, clinical_data)
        patient_dir=os.path.dirname(notebook_path)
        # Convert the notebook to Markdown
        output_md = self.convert_notebook_to_md(notebook_path)
        # Read the markdown content
        with open(output_md, 'r', encoding='utf-8') as file:
            md_content = file.read()
        md_content = self.embed_images_in_markdown(md_content, patient_dir)
        

        messages = [
            (
                "system",
                str(self.instructions)
            ),
            (
                "human",
                md_content
            )
        ]
        # Run the model
        response = self.run_model(messages)
        self.log_message("Predictor node response: " + response.content[:100])
        state = state | {
            'patient_data' : md_content,
            'response' : md_content + response.content,
            'answer' : 'yes',
            'answer_source':'predictor'

        }
        return state
    


class RagNode(node):
    """
    A LangGraph node for Retrieval Augmented Generation using a pre-built index.
    Inherits from the base 'node' class.
    """
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None):
        super().__init__(llm=llm, instructions=instructions, functions=functions, welcome=welcome, logging_key="RAG node.- ")
        self.llm_rag = ChatGoogleGenerativeAI(model=self.llm,
                                               temperature=0.5,)

    def get_retriever(self, vector_store, k) -> Runnable:
        """
        Retrieves the vector store from the state and returns it as a retriever.
        This method is expected to be overridden in subclasses if needed.
        """
        if not vector_store:
            raise ValueError("Vector store not found in state.")
        return vector_store.as_retriever(search_kwargs={"k": k})
    
    def run_model(self, messages):
        """Run the model with the given messages."""
        response = self.llm_rag.invoke(messages)
        return response
    
    def build_sources_list(self, response) -> str:
        """
        Extracts sources from the response and returns them as a set.
        """
        sources = []
        for document in response['context']:
            if 'source' in document.metadata:
                file=os.path.basename(document.metadata['source'])
                sources.append(file)
            else:
                continue
        sources=set(sources)
        bibliography=""
        for source in sources:
            bibliography += f"- [{source}](https://vergaju.github.io/data_techtask/{source})\n"
        return bibliography

    def get_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get RAG model response based on the provided state.
        """
        vector_store = state.get('vector_db')
        k = state.get('k_retrieval', 5)  # Default to 5 if not specified
        retriever = self.get_retriever(vector_store, k)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.instructions),
            ("human", "{input}")
        ])
        combine_docs_chain = create_stuff_documents_chain(
            self.llm_rag,
            prompt_template
        )
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        response = retrieval_chain.invoke({"input": state.get("messages", "")})
        bibliography = self.build_sources_list(response)

        final_response = response.get('answer', '') + "\n\n" + bibliography

        self.log_message(f"RAG node response: {response}")
        state = state | {
            'response' : final_response,
            'answer' : 'yes',
            'answer_source':'RAG'

        }
        return state
    

class LiteratureNode(node):
    """
    A LangGraph node for Literature Retrieval using a pre-built index.
    Inherits from the base 'node' class.
    """
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None):
        super().__init__(llm=llm, instructions=instructions, functions=functions, welcome=welcome, logging_key="Literature node.- ")
        self.llm_literature = ChatGoogleGenerativeAI(model=self.llm,
                                               temperature=0.0,)
    
        self.set_config()

    def set_config(self):
        self.config_with_search = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            )

    def run_model(self, user_query):
        """Run the model with the given messages."""
        logging.info(f"Running model with user query: {user_query}")
        response = self.client.models.generate_content(
            model=self.llm,
            contents=user_query,
            config=self.config_with_search,
        )
        return response
    
    def format_text(self, response):
        answer = response.text
        chunks = response.candidates[0].grounding_metadata.grounding_chunks
        supports = response.candidates[0].grounding_metadata.grounding_supports
        research_queries=response.candidates[0].grounding_metadata.web_search_queries
        lit_tools_instance = self.functions(chunks, supports, answer)

        answer=lit_tools_instance.process_references()
        bibliography=lit_tools_instance.create_bibliography()

        return answer, bibliography, research_queries
    
    def get_node(self, state):
        """Perform GroundSearch with UserQuery
        Returns:
        - Answer: Markdown formatted text answer from Ground Search iwth clickable references
        - Bibliography: References, link and website use to obtain the answer
        - ResearchQueries: Queries used to perform GroundSearch"""
        user_query = state['messages']
        logging.info(f"Searching {user_query} with GroundSearch model")

        logging.info("Performing GroundSearch")
        question = self.instructions + user_query

        response = self.run_model(question)
        logging.info(f"GroundSearch Response: {response}")

        answer,bibliography, research_queries= self.format_text(response)

        response = answer + bibliography + "\n\n" + "Research queries used:\n" + "\n".join(research_queries)
        
        state = state | {
            'response' : response,
            'answer' : 'yes',
            'answer_source':'GroundSearch'

        }
        return state
    
        