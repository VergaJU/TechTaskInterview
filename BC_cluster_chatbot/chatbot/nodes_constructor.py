from langchain_core.messages import ( 
    AIMessage
    )
from google.genai.types import GenerateContentResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai import types
from google import genai 
from dotenv import load_dotenv
import os
import papermill as pm
import subprocess
import base64
from bs4 import BeautifulSoup
import io
import re
import json
# save logs
import logging

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
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None, complete_answer=None):
        super().__init__(llm=llm, instructions=instructions, functions=functions, welcome=welcome, logging_key="Master node.- ")
        self.llm_master=ChatGoogleGenerativeAI(model=self.llm)


    def set_model(self, model):
        self.llm_master = ChatGoogleGenerativeAI(model=model)

    def run_model(self, messages):
        """Run the model with the given messages."""
        #print(f"--- Message going to the llm_master: {messages}---")
        response = self.llm_master.invoke(str(self.instructions) + messages)
        return response
    
    def get_node(self, state):
        """The master node, which is the first node in the graph. It's role is to route the workflow to specific nodes and interface the user"""
        messages = state['messages']
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
        session_dir = os.path.join('/app', session_id)
        os.makedirs(session_dir, exist_ok=True)

        notebook_path = '/app/notebooks/Sample_report.ipynb'
        output_path = os.path.join(session_dir, 'Sample_report_' + session_id + '.ipynb')
        pm.execute_notebook(
            input_path=notebook_path,
            output_path=output_path,
            parameters={
                'sample': patient_expression,
                'clinical_data': clinical_data
            }
        )
        return output_path
    
    def convert_notebook_to_html(self, notebook_path):
        """Convert the notebook to HTML."""
        output_html = notebook_path.replace('.ipynb', '.html')
        subprocess.run([
            "jupyter", "nbconvert", "--to", "html", "--no-input",
            notebook_path, "--output", output_html
        ])
        return output_html

    def parse_html(self, output_html):
        """Parse the HTML content and extract the relevant information."""

        clean_text=BeautifulSoup(html_content, 'html.parser').get_text()
        clean_text = base64.b64encode(clean_text).decode('utf-8')

        return clean_text


    def set_model(self, model):
        self.llm_predictor = ChatGoogleGenerativeAI(model=model)

    def run_model(self, clean_text):
        """Run the model with the given messages."""
        response = self.llm_predictor.invoke(str(self.instructions) + clean_text)
        return response

    def get_node(self,state):
        """The predictor node, which is the second node in the graph. It's role is to predict the cluster of the patient and convert the notebook into an html report"""
        expression_data = state['expression_data']
        clinical_data = state['clinical_data']
        # Compile the notebook
        session_id = state['session_id']
        notebook_path = self.compile_notebook(session_id, expression_data, clinical_data)
        # Convert the notebook to HTML
        output_html = self.convert_notebook_to_html(notebook_path)
        with open(output_html, 'rb') as f:
            html_content = f.read()
        # Parse the HTML content
        clean_text = self.parse_html(html_content)

        # Run the model
        response = self.run_model(clean_text)
        self.log_message("Predictor node response: " + response)
        state = state | {
            'patient_data' : html_content,
            'response' : response.content,
            'answer' : 'yes',
            'answer_source':'predictor'

        }
        return state




