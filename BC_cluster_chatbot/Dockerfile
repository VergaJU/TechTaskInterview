FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN apt-get update
RUN pip install scanpy scikit-learn urllib3 optuna
RUN pip install jupyter lifelines igraph leidenalg shap gseapy lxml
RUN pip install streamlit
RUN pip install "langgraph==0.3.21" "google-genai==0.8.0" "langchain==0.3.24" "langchain-community==0.3.22" "langchain-core==0.3.56" "langchain-google-genai==2.1.2" "langchain-text-splitters==0.3.8" "google-genai==0.8.0"
RUN pip install python-dotenv papermill

# Set environment variables to avoid buffer issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
EXPOSE 8501
CMD ["streamlit", "run", "/app/BC_cluster_chatbot/app.py", "--server.port=8501", "--server.address=0.0.0.0"]