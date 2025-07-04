FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN apt-get update
RUN pip install scanpy scikit-learn urllib3 optuna
RUN pip install jupyter lifelines igraph leidenalg shap
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]