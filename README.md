<div align="left" style="position: relative;">
<img src="./asset/logo.svg" align="right" width="30%" style="margin: 100px 0 0 20px;">
<h1>TechTaskInterview</h1>
<p align="left">
	<em><code>❯ Repository including the code for prearing to the interview with Dr. Sebastiano Panichella and Dr. Saša Miladinović </code></em>
</p>
<p align="left">
	<img src="https://img.shields.io/github/license/VergaJU/TechTaskInterview?style=default&logo=opensourceinitiative&logoColor=white&color=3fff00" alt="license">
	<img src="https://img.shields.io/github/last-commit/VergaJU/TechTaskInterview?style=default&logo=git&logoColor=white&color=3fff00" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/VergaJU/TechTaskInterview?style=default&color=3fff00" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/VergaJU/TechTaskInterview?style=default&color=3fff00" alt="repo-language-count">
</p>
<p align="left"><!-- default option, no dependency badges. -->
</p>
<p align="left">
	<!-- default option, no dependency badges. -->
</p>
</div>
<br clear="right">

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

Task:Bioinformatics Development & Analysis Task (AI Integration)

Dataset link: [GSE96058](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058)

Instruction:
- Download and preprocess data
- Perform basic quality control and normalization where needed
- Explore patterns related to disease subtypes, treatment response, or other relevant biological variable

Analysis:
-  Apply one or more AI techniques such as:
    - Unsupervised learning (e.g., clustering to find molecular subtypes).
    - Supervised learning (e.g., building classifiers for clinical labels).
    - LLM integration (e.g., generating biological interpretations or labeling clusters).
    - etc.

Expected Output:

- A summary of your analysis and results.
- Insights into the biological or clinical relevance of the patterns you discovered.
- Reflections on the performance and interpretability of the AI methods used.

Deliverables:

- A short technical presentation (slides) summarizing your methodology, models, results, and interpretations.
- Supporting materials such as source code, model artifacts, plots, or notebooks.
- Deliverables are expected shortly before the interview (slides and code should be shared the day before the interview); timing will be agreed upon individually.


##  Features


Analysis:
- Downloaded $log_2(FPKM)$
- Averaged expression of replicates
- Trained Autoencoder to obtain representative embeddings (**Supervided Learning**)
- Clustered samples using leiden clustering and neighbrohood graph with the Python packge Scanpy (**Unsupervised Learning**)
- Clusters annotaton considering clinical features (PAM50, NHG), gene module scoring and GSEA (**Bioinformatics**)
    - Performed DGE with Scanpy (method='wilcoxon') to rank gene expression
    - Ran GSEA using the prerank function from the python package gseapy
- Fine tuning classifier using the already trained encoder and a classification head to predict the annotated clusters
    - Frozen encoder weights
    - Training classification head
- LLM Chatbot (Gemini + LangGraph) (**LLM integration**):
    - Router node orchestrate needed nodes
    - Predictor node classify patient dependinng on gene expression and provide detailed report including SHAP values, GSEA and clinical features.
    - RAG provide detailed information regarding Clusters annotation and Cox regression
    - Literature node perform GroundSearch on scientific literature regarding Breast Cancer

---

##  Project Structure

```sh
└── TechTaskInterview/
    ├── AE
    │   ├── AE.py
    │   ├── AEclassifier.py
    │   └── __init__.py
    ├── BC_cluster_chatbot
    │   ├── app.py
    │   ├── chatbot
    │   │   ├── __init__.py
    │   │   ├── create_db.py
    │   │   ├── graph_state.py
    │   │   ├── literature_functions.py
    │   │   ├── nodes.py
    │   │   ├── nodes_constructor.py
    │   │   ├── prompts.py
    │   │   ├── prompts.yaml
    │   │   └── workflow.py
    │   └── test_expression.csv
    ├── LICENSE
    ├── README.md
    ├── autoencoder_params.yaml
    ├── classifier_params.yaml
    ├── compose.yaml
    ├── containers
    │   ├── chatbot
    │   │   └── Dockerfile
    │   ├── jupyter_env
    │   │   └── Dockerfile
    │   └── torch_env
    │       └── Dockerfile
    ├── corpus
    │   ├── Basal-G3.html
    │   ├── Her2-G2.html
    │   ├── Her2-G3.html
    │   ├── LumA-G2.html
    │   ├── LumB-G3.html
    │   ├── Normal-G2.html
    │   └── Regression_clusters.html
    ├── logs
    │   ├── classifier_metrics.pkl
    │   └── classifier_training_logs.csv
    ├── models
    │   ├── SHAP.pkl
    │   ├── autoencoder_model.pth
    │   ├── classifier_model.pth
    │   └── standard_scaler.pkl
    ├── notebooks
    │   ├── Basal-G3.ipynb
    │   ├── Classifier_metrics.ipynb
    │   ├── Cluster_annotation.ipynb
    │   ├── Clustering_1.ipynb
    │   ├── Get_data.ipynb
    │   ├── Her2-G2.ipynb
    │   ├── Her2-G3.ipynb
    │   ├── LumA-G2.ipynb
    │   ├── LumB-G3.ipynb
    │   ├── Normal-G2.ipynb
    │   ├── Regression_clusters.ipynb
    │   ├── Sample_report.ipynb
    │   ├── Untitled.ipynb
    │   ├── Untitled1.ipynb
    │   └── train_data.ipynb
    └── scripts
        ├── get_data.py
        ├── get_embeddings.py
        ├── make_shap.py
        ├── prepare_train_data.py
        ├── prepare_train_data_classifier.py
        ├── train.py
        └── train_classifier.py
```


###  Project Index
<details open>
	<summary><b><code>TECHTASKINTERVIEW/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/compose.yaml'>compose.yaml</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/classifier_params.yaml'>classifier_params.yaml</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/autoencoder_params.yaml'>autoencoder_params.yaml</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- BC_cluster_chatbot Submodule -->
		<summary><b>BC_cluster_chatbot</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/BC_cluster_chatbot/app.py'>app.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
			<details>
				<summary><b>chatbot</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/BC_cluster_chatbot/chatbot/nodes_constructor.py'>nodes_constructor.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/BC_cluster_chatbot/chatbot/prompts.yaml'>prompts.yaml</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/BC_cluster_chatbot/chatbot/workflow.py'>workflow.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/BC_cluster_chatbot/chatbot/graph_state.py'>graph_state.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/BC_cluster_chatbot/chatbot/nodes.py'>nodes.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/BC_cluster_chatbot/chatbot/prompts.py'>prompts.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/BC_cluster_chatbot/chatbot/create_db.py'>create_db.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/BC_cluster_chatbot/chatbot/literature_functions.py'>literature_functions.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- scripts Submodule -->
		<summary><b>scripts</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/scripts/prepare_train_data.py'>prepare_train_data.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/scripts/prepare_train_data_classifier.py'>prepare_train_data_classifier.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/scripts/get_data.py'>get_data.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/scripts/get_embeddings.py'>get_embeddings.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/scripts/train_classifier.py'>train_classifier.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/scripts/train.py'>train.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/scripts/make_shap.py'>make_shap.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- AE Submodule -->
		<summary><b>AE</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/AE/AE.py'>AE.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/AE/AEclassifier.py'>AEclassifier.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- notebooks Submodule -->
		<summary><b>notebooks</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/Basal-G3.ipynb'>Basal-G3.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/Normal-G2.ipynb'>Normal-G2.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/Untitled.ipynb'>Untitled.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/Get_data.ipynb'>Get_data.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/Her2-G3.ipynb'>Her2-G3.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/Her2-G2.ipynb'>Her2-G2.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/Sample_report.ipynb'>Sample_report.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/Classifier_metrics.ipynb'>Classifier_metrics.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/Untitled1.ipynb'>Untitled1.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/LumA-G2.ipynb'>LumA-G2.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/Cluster_annotation.ipynb'>Cluster_annotation.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/Regression_clusters.ipynb'>Regression_clusters.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/LumB-G3.ipynb'>LumB-G3.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/Clustering_1.ipynb'>Clustering_1.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/notebooks/train_data.ipynb'>train_data.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- containers Submodule -->
		<summary><b>containers</b></summary>
		<blockquote>
			<details>
				<summary><b>chatbot</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/containers/chatbot/Dockerfile'>Dockerfile</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>torch_env</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/containers/torch_env/Dockerfile'>Dockerfile</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>jupyter_env</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/containers/jupyter_env/Dockerfile'>Dockerfile</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- corpus Submodule -->
		<summary><b>corpus</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/corpus/LumA-G2.html'>LumA-G2.html</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/corpus/LumB-G3.html'>LumB-G3.html</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/corpus/Her2-G2.html'>Her2-G2.html</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/corpus/Normal-G2.html'>Normal-G2.html</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/corpus/Regression_clusters.html'>Regression_clusters.html</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/corpus/Basal-G3.html'>Basal-G3.html</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/corpus/Her2-G3.html'>Her2-G3.html</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- models Submodule -->
		<summary><b>models</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/models/classifier_model.pth'>classifier_model.pth</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/VergaJU/TechTaskInterview/blob/master/models/autoencoder_model.pth'>autoencoder_model.pth</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with TechTaskInterview, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Container Runtime:** Docker


###  Installation

Install TechTaskInterview using one of the following methods:

**Build from source:**

1. Clone the TechTaskInterview repository:
```sh
❯ git clone https://github.com/VergaJU/TechTaskInterview
```

2. Navigate to the project directory:
```sh
❯ cd TechTaskInterview
```

3. Install the project dependencies:


**Using `docker`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Docker-2CA5E0.svg?style={badge_style}&logo=docker&logoColor=white" />](https://www.docker.com/) and compose:

```sh
❯ docker compose --profile torch_gpu build
❯ docker compose --profile jupyter build
❯ docker compose --profile chatbot build
```




###  Usage

#### Data and model preparation:

Prepare data, train autoencoder and obtain embeddings with:

```sh
❯ docker compose --profile torch_gpu up -d
❯ docker exec -it torch_gpu bash
❯ python scripts/get_data.py
❯ python scripts/prepare_train_data.py
❯ python scripts/train.py

```


#### Training metrics and clusters annotation:

```sh
❯ docker compose --profile jupyter up -d
```


Go to `localhost:8888` and run the notebooks:
- Classifier_metrics.ipynb
- Clustering_1.ipynb
- Regression_clusters.ipynb
- Cluster specific notebooks

#### Training classifier:

```sh
❯ docker compose --profile torch_gpu up -d
❯ docker exec -it torch_gpu bash
❯ python scripts/prepare_train_data_classifier.py
❯ python scripts/train_classifier.py
❯ python scripts/make_shap.py
```


#### Chatbot:


```sh
❯ docker compose --profile chatbot up -d
```

Go to `localhost:8890` and anjoy the notebook


###  Testing
Testing not implemented yet

---
##  Project Roadmap

- [X] **`Data`**: <strike>Imported correctly data.</strike>
- [X] **`Analysis`**: <strike>Completed Bioinformatic analysis.</strike>.
- [X] **`AI`**: <strike>Completed AI integration.</strike>.
- [ ] **`Testing`**: Implement tests

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/VergaJU/TechTaskInterview/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/VergaJU/TechTaskInterview/issues)**: Submit bugs found or log feature requests for the `TechTaskInterview` project.
- **💡 [Submit Pull Requests](https://github.com/VergaJU/TechTaskInterview/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/VergaJU/TechTaskInterview
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/VergaJU/TechTaskInterview/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=VergaJU/TechTaskInterview">
   </a>
</p>
</details>

---

##  License

This project is protected under the [Apache License 2.0](https://choosealicense.com/licenses) License. For more details, refer to the [Apache License 2.0](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---