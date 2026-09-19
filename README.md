# Graph-Based Cyber Threat Analysis for Entity Classification and Threat Association Detection



This repository contains the complete implementation of a graph-centric Cyber Threat Intelligence (CTI) research project. The workflow starts from a **pre-annotated CTI dataset**, constructs and analyses a directed Knowledge Graph, refines duplicate entities through canonicalisation, learns Node2Vec graph embeddings, compares traditional machine-learning models with GraphSAGE for entity-type classification, performs a final comparative and statistical evaluation, and finishes with a secondary directed threat-association detection experiment.

The implementation is organised into **nine Jupyter notebooks**. They should be run in numerical order because later notebooks use files produced by earlier notebooks.

```text
Annotated CTI dataset
        |
        v
01  Data exploration
        |
        v
02  Knowledge Graph construction
        |
        v
03  Graph analytics
        |
        v
04  Entity canonicalisation and graph refinement
        |
        v
05  Node2Vec graph embeddings
        |
        v
06  Traditional ML baseline
        |
        v
07  GraphSAGE node classification
        |
        v
08  Final comparative evaluation
        |
        v
09  Threat-association detection
```

---

# 1. Project purpose

The project investigates whether improving the structure of a Cyber Threat Intelligence Knowledge Graph can improve downstream graph-learning performance.

## 1.1 Research questions

The final implementation addresses two primary research questions:

**RQ1 — Knowledge Graph refinement**

> Can Knowledge Graph refinement through entity canonicalisation improve graph representation learning and downstream cyber entity classification?

**RQ2 — Graph Neural Network versus traditional machine learning**

> Can a Graph Neural Network that directly aggregates neighbourhood information outperform traditional machine learning models based on fixed Node2Vec embeddings?

Notebook 08 consolidates the evidence for both research questions. Notebook 09 is a secondary directed threat-association detection extension and is not treated as a third primary research question.

The main workflow is:

1. inspect and validate the supplied CTI dataset;
2. build a directed CTI Knowledge Graph from annotated entities and relations;
3. analyse graph structure, sparsity, connectivity and important nodes;
4. reduce graph fragmentation using entity canonicalisation;
5. learn 128-dimensional Node2Vec structural embeddings;
6. train traditional machine-learning baselines for entity-type classification;
7. train and evaluate a direction-aware GraphSAGE classifier;
8. compare full-graph, connected-node and original-versus-canonical conditions using repeated runs and statistical tests;
9. perform a secondary directed threat-association detection experiment.

The principal research task is **transductive multiclass node classification**. Notebook 09 is a secondary **directed binary association-detection** experiment.

---

# 2. What the dataset contains

The project uses a **pre-annotated CTI dataset**. The entity and relation annotations already exist in the source data; this project does not perform named-entity extraction from raw text as its main task.

The submitted dataset contains approximately:

- **9,732 CTI reports**;
- **20 entity types**;
- **25 relation types**;
- **12,679 mention-level graph nodes** before canonicalisation;
- **1,094 directed graph relations** before canonicalisation.

Examples of entity types include:

- `DOMAIN`
- `EMAIL`
- `FILEPATH`
- `IPV4`
- `SOFTWARE`
- `URL`
- `attack-pattern`
- `campaign`
- `identity`
- `location`
- `malware`
- `threat-actor`
- `tools`
- `vulnerability`

Examples of relation types include:

- `uses`
- `targets`
- `indicates`
- `related-to`
- `located-at`
- `exploits`
- `attributed-to`
- `delivers`
- `downloads`
- `communicates-with`

---

# 3. Project folder structure

After extracting the submission ZIP, locate the folder that directly contains:

```text
data/
notebooks/
outputs/
requirements.txt
README.md
```

That folder is called the **project root** throughout this README and is written as:

```text
<PROJECT_ROOT>
```

`<PROJECT_ROOT>` is a placeholder. Do not type it literally into commands that require an actual filesystem location.

The supplied project structure is:

```text
<PROJECT_ROOT>/
|
|-- data/
|   `-- archive/
|       |-- all.jsonl
|       |-- Cyber-Threat-Intelligence-Custom-Data_new_processed.csv
|       |-- cyber-threat-intelligence-splited_test.csv
|       |-- cyber-threat-intelligence-splited_train.csv
|       |-- cyber-threat-intelligence-splited_validate.csv
|       |-- cyber-threat-intelligence_all.csv
|       |-- test.jsonl
|       |-- train.jsonl
|       `-- validation.jsonl
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_knowledge_graph_construction.ipynb
|   |-- 03_graph_analytics.ipynb
|   |-- 04_entity_canonicalisation_graph_refinement.ipynb
|   |-- 05_graph_embeddings_node2vec.ipynb
|   |-- 06_traditional_ml_baseline.ipynb
|   |-- 07_graphsage_node_classification.ipynb
|   |-- 08_final_comparative_evaluation.ipynb
|   `-- 09_threat_association_detection_FINAL.ipynb
|
|-- outputs/
|   |-- node2vec_embeddings/
|   |-- canonicalisation_v4_manual_validation_sample.csv
|   |-- canonical_entity_mapping_v2.csv
|   |-- canonical_entity_mapping_v3.csv
|   |-- canonical_entity_mapping_v4.csv
|   |-- canonical_merge_examples.csv
|   |-- community_entity_type_summary.csv
|   |-- community_summary.csv
|   |-- cti_knowledge_graph.pkl
|   |-- cti_knowledge_graph_canonical_final.pkl
|   |-- cti_knowledge_graph_canonical_v2.pkl
|   |-- cti_knowledge_graph_canonical_v3.pkl
|   |-- dataset_role_definition.csv
|   |-- degree_distribution.png
|   |-- entity_type_distribution.png
|   |-- graph_basic_statistics.csv
|   |-- graph_betweenness_scores.csv
|   |-- graph_combined_importance_scores.csv
|   |-- graph_component_analysis.csv
|   |-- graph_degree_analysis.csv
|   |-- graph_important_entities_by_type.csv
|   |-- graph_original_vs_canonical_v1_v2_v3_v4_comparison.csv
|   |-- graph_pagerank_scores.csv
|   |-- graph_relation_type_distribution.csv
|   |-- kg_canonical_entities_v2.csv
|   |-- kg_canonical_entities_v3.csv
|   |-- kg_canonical_entities_v4.csv
|   |-- kg_canonical_relations_v2.csv
|   |-- kg_canonical_relations_v3.csv
|   |-- kg_canonical_relations_v4.csv
|   |-- kg_edge_type_distribution.csv
|   |-- kg_entities.csv
|   |-- kg_graph_summary.csv
|   |-- kg_node_edge_type_distribution.png
|   |-- kg_node_type_distribution.csv
|   |-- kg_relations_all.csv
|   |-- kg_relations_valid.csv
|   |-- largest_community_relation_summary.csv
|   |-- node_communities.csv
|   |-- relation_type_distribution.png
|   |-- report_length_distribution.png
|   `-- top_nodes_per_community.csv
|
`-- requirements.txt
```

The `outputs/node2vec_embeddings/` directory contains:

```text
outputs/node2vec_embeddings/
|-- figures/
|   |-- node2vec_pca_visualisation.png
|   |-- node2vec_training_loss.png
|   `-- node2vec_tsne_visualisation.png
|-- node2vec_embeddings.npy
|-- node2vec_embeddings_with_metadata.csv
|-- node2vec_embedding_quality_summary.csv
|-- node2vec_link_prediction_evaluation.csv
|-- node2vec_loss_history.csv
|-- node2vec_model_state_dict.pt
|-- node2vec_nearest_neighbour_example.csv
|-- node2vec_training_summary.csv
|-- node_index_mapping.csv
`-- node_metadata.csv
```

When Notebooks 06-09 are executed, they also create additional result folders under `outputs/`, including:

```text
outputs/ml_baseline/
outputs/graphsage_final/
outputs/final_evaluation/
outputs/threat_association_detection/
```

These folders contain the final model files, split assignments, repeated-seed results, statistical summaries and dissertation-ready figures.

---

# 4. Before starting

## 4.1 Recommended environment

The final graph-learning environment used:

```text
Python:                   3.11.15
PyTorch:                  2.12.1+cpu
PyTorch Geometric:        2.8.0
scikit-learn:             1.7.2
Execution device:         CPU
GPU:                      not required
```

A GPU is optional. The submitted workflow is designed to run on CPU.

Some earlier notebook metadata records a historical Python 3.14 kernel, while the final graph-learning notebooks were executed in the dedicated Python 3.11 environment above. For a clean end-to-end reproduction, use **Python 3.11 for all nine notebooks** rather than mixing kernels.

## 4.2 Main Python packages

The supplied `requirements.txt` installs the normal project dependencies, including:

- JupyterLab;
- ipykernel and ipywidgets;
- NumPy;
- pandas;
- Matplotlib;
- NetworkX;
- SciPy;
- scikit-learn;
- joblib;
- tqdm;
- PyTorch Geometric.

For the submitted CPU configuration, **PyTorch 2.12.1 is installed first using the official PyTorch CPU index**, and **`pyg_lib` is installed afterwards from the matching PyTorch Geometric wheel page**. These two packages are intentionally installed separately because they use specialised binary-wheel repositories.

The final requirements configuration is intended for the Windows/Linux CPU environment used by the project.

---

# 5. Extracting the ZIP file

This section is written for someone with little or no Python experience.

Assume the downloaded submission is a ZIP archive such as:

```text
cti-graph-project.zip
```

The exact ZIP filename does not matter.

## 5.1 Windows File Explorer

1. Locate the ZIP file in **File Explorer**.
2. Right-click the ZIP file.
3. Select **Extract All...**.
4. Choose any folder where you have read/write permission.
5. Click **Extract**.
6. Open the extracted folder.
7. Continue opening folders until you see `data`, `notebooks`, `outputs`, `requirements.txt` and `README.md` together.
8. That folder is the **project root**.

Do not move individual notebooks away from the project structure. The relative relationship between `notebooks/`, `data/` and `outputs/` is important.

## 5.2 Windows PowerShell alternative

If the ZIP file is in the current PowerShell directory, it can also be extracted using:

```powershell
Expand-Archive -Path .\cti-graph-project.zip -DestinationPath .\cti-graph-project -Force
```

After extraction, enter the folder that directly contains `data`, `notebooks`, `outputs` and `requirements.txt`.

## 5.3 macOS or Linux alternative

If the `unzip` utility is available:

```bash
unzip cti-graph-project.zip -d cti-graph-project
```

Then enter the extracted project root.

---

# 6. Confirm the required data files

Before installing or running anything, confirm that the following files exist inside:

```text
<PROJECT_ROOT>/data/archive/
```

Required files:

```text
all.jsonl
train.jsonl
validation.jsonl
test.jsonl
cyber-threat-intelligence_all.csv
cyber-threat-intelligence-splited_train.csv
cyber-threat-intelligence-splited_validate.csv
cyber-threat-intelligence-splited_test.csv
Cyber-Threat-Intelligence-Custom-Data_new_processed.csv
```

`all.jsonl` is the principal source used for Knowledge Graph construction and canonicalisation.

The report-level `train.jsonl`, `validation.jsonl` and `test.jsonl` files are explored during the data stage. The later node-classification experiment uses a separate fixed **node-level** train/validation/test split created in Notebook 06.

If any required dataset file is missing, restore it to `data/archive/` before running Notebook 01.

---

# 7. Recommended installation: Conda

For a non-technical reproducer, using a fresh Conda environment is recommended because it keeps this project's Python packages separate from unrelated software on the computer.

## 7.1 Install Conda if necessary

Install either **Miniconda** or **Anaconda** using the official installer for the operating system.

After installation, open **Anaconda Prompt** or a terminal where `conda` is available.

Check that Conda works:

```bash
conda --version
```

A version number should be displayed.

## 7.2 Open a terminal in the project root

The terminal's current directory must be the extracted project root — the folder containing:
```text
Anaconda is prferred
Navigating to Project directory
cd /d <Your project path>

For eg:  cd /d D:\cti-graph-project
Make sure the the following cmds are running in conda till step 7.7 

```
```text
data/
notebooks/
outputs/
requirements.txt
README.md
```

On Windows, a simple method is:

1. open the project root in File Explorer;
2. click the address bar;
3. type `powershell`;
4. press Enter.

If using Anaconda Prompt, use `cd` to move to the project root.

## 7.3 Create a clean Python 3.11 environment

```bash
conda create -n cti-graph-ml python=3.11 -y
```

Activate it:

```bash
conda activate cti-graph-ml
```

Confirm the Python version:

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

## 7.4 Upgrade package-management tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 7.5 Install the project dependencies

From the project root, use the following commands **in this order**.

### Step 1 — Install the CPU build of PyTorch used by the final experiments

```bash
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
```

### Step 2 — Install the normal project dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 3 — Install `pyg_lib` for Notebook 05 Node2Vec

```bash
python -m pip install pyg_lib -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
```

The `torch-2.12.0+cpu` wheel page is intentional for the PyTorch **2.12.x** family, including the submitted PyTorch 2.12.1 CPU environment.

Installation may take several minutes because PyTorch and graph-learning packages are relatively large.

## 7.6 Verify the important libraries

Run:

```bash
python -c "import torch, torch_geometric, pyg_lib, sklearn, numpy, pandas, networkx, scipy; print('Python environment OK'); print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('scikit-learn:', sklearn.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For the submitted CPU environment, `CUDA available: False` is normal.

## 7.7 Register the environment as a Jupyter kernel

```bash
python -m ipykernel install --user --name cti-graph-ml --display-name "Python 3.11 (CTI Graph Project)"
```

This creates a Jupyter kernel named:

```text
Python 3.11 (CTI Graph Project)
```

Use this kernel for all nine notebooks.

GO to step 9 after setting up the kernel

---

# 8. Alternative installation using Python venv

If Conda cannot be used but Python 3.11 is already installed, a standard virtual environment can be created instead.

## 8.1 Windows

From the project root:

```powershell
py -3.11 -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell prevents script activation, Command Prompt can activate the same environment with:

```cmd
venv\Scripts\activate.bat
```

Then run:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install pyg_lib -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
python -m ipykernel install --user --name cti-graph-ml --display-name "Python 3.11 (CTI Graph Project)"
```

## 8.2 Linux

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install pyg_lib -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
python -m ipykernel install --user --name cti-graph-ml --display-name "Python 3.11 (CTI Graph Project)"
```

## 8.3 macOS

The submitted experiments used the Windows/Linux CPU wheel configuration. On macOS, create a Python 3.11 environment but install the platform-appropriate PyTorch and PyTorch Geometric builds using their official installation instructions rather than copying the Windows/Linux CPU wheel commands verbatim. The notebook code itself does not depend on Windows-specific APIs, except that Notebook 09's `PROJECT_ROOT` value must be changed to the extracted project location.


The `venv/` folder is machine-specific and normally should not be included in a submission ZIP or source-control repository.

---

# 9. Start JupyterLab correctly

The working directory matters.

**Start JupyterLab from the `notebooks/` folder.**

The notebooks use relative paths such as:

```python
Path("..")
Path("../data/archive")
Path("../outputs")
Path.cwd().resolve().parent
```

Starting JupyterLab from the wrong folder can cause a `FileNotFoundError` even when the files exist.

From the project root:

```bash
cd notebooks
```

Then start JupyterLab:

```bash
python -m jupyter lab
```

A browser window should open automatically.

In JupyterLab, confirm that the selected kernel is:

```text
Python 3.11 (CTI Graph Project)
```

If necessary, use the JupyterLab **Kernel** menu to change the kernel.

To confirm the environment from inside a notebook, run:

```python
import sys
print(sys.version)
print(sys.executable)
```

The Python version should be 3.11.x and the executable should belong to the environment created for this project.

---

# 10. Important path check for Notebook 09

Notebooks 01-08 are designed around the project folder relationship described above.

Before running Notebook 09, inspect its initial configuration/path cell and find the `PROJECT_ROOT` setting.

`PROJECT_ROOT` must point to the folder that directly contains:

```text
data/
notebooks/
outputs/
requirements.txt
```

The submitted Notebook 09 uses a fixed absolute `PROJECT_ROOT` value. It is correct for the original completed run.

For reproduction on another computer, replace only that value with the absolute location of the extracted project root.

Conceptually, it should be equivalent to:

```python
PROJECT_ROOT = Path(r"<PROJECT_ROOT>")
```

where `<PROJECT_ROOT>` must be replaced by the actual extracted project-root path. Do not enter the angle-bracket placeholder literally.

Do not change the experiment settings merely because the project was extracted to a different folder.

---

# 11. Exact notebook execution order

Run the notebooks in this exact order:

```text
01_data_exploration.ipynb
        |
        v
02_knowledge_graph_construction.ipynb
        |
        v
03_graph_analytics.ipynb
        |
        v
04_entity_canonicalisation_graph_refinement.ipynb
        |
        v
05_graph_embeddings_node2vec.ipynb
        |
        v
06_traditional_ml_baseline.ipynb
        |
        v
07_graphsage_node_classification.ipynb
        |
        v
08_final_comparative_evaluation.ipynb
        |
        v
09_threat_association_detection_FINAL.ipynb
```

Do not run the notebooks in an arbitrary order when reproducing the project from scratch.

For every notebook:

1. open the notebook in JupyterLab;
2. confirm `Python 3.11 (CTI Graph Project)` is selected;
3. choose **Restart Kernel and Run All Cells**;
4. wait until execution finishes;
5. scroll through the notebook and check that no red traceback/error remains;
6. confirm the expected output/checkpoint file exists;
7. only then continue to the next notebook.

Do not delete `outputs/` between notebooks. Later stages use files created by earlier stages.

---

# 12. Quick checkpoint table

Use this table to verify that each stage completed successfully.

| Notebook | Minimum checkpoint to confirm |
|---|---|
| 01 | `outputs/dataset_role_definition.csv` |
| 02 | `outputs/cti_knowledge_graph.pkl` |
| 03 | `outputs/graph_basic_statistics.csv` |
| 04 | `outputs/cti_knowledge_graph_canonical_final.pkl` |
| 05 | `outputs/node2vec_embeddings/node2vec_embeddings_with_metadata.csv` |
| 06 | `outputs/ml_baseline/node_classification_split_assignments.csv` |
| 07 | `outputs/graphsage_final/final_experiment_configuration.json` |
| 08 | `outputs/final_evaluation/final_research_summary.json` |
| 09 | `outputs/threat_association_detection/detection_experiment_configuration.json` |

If a checkpoint is missing, stop and resolve the corresponding notebook before moving forward.

---

# 13. Notebook-by-notebook guide

## 13.1 Notebook 01 — Data Exploration

File:

```text
notebooks/01_data_exploration.ipynb
```

### Purpose

This notebook establishes what is present in the dataset before any graph model is built.

It:

- loads the supplied JSONL and CSV files;
- checks the dataset schema;
- counts reports, entities and relations;
- inspects entity-type and relation-type distributions;
- checks report lengths and missing/malformed values;
- records the role of each dataset file;
- generates basic exploratory figures.

### Main inputs

```text
data/archive/all.jsonl
data/archive/train.jsonl
data/archive/validation.jsonl
data/archive/test.jsonl
data/archive/cyber-threat-intelligence_all.csv
data/archive/cyber-threat-intelligence-splited_train.csv
data/archive/cyber-threat-intelligence-splited_validate.csv
data/archive/cyber-threat-intelligence-splited_test.csv
data/archive/Cyber-Threat-Intelligence-Custom-Data_new_processed.csv
```

### Main outputs

```text
outputs/dataset_role_definition.csv
outputs/report_length_distribution.png
outputs/entity_type_distribution.png
outputs/relation_type_distribution.png
```

### What success looks like

The notebook should finish without errors and report the expected CTI dataset structure. The full dataset contains approximately 9,732 reports, 20 entity types and 25 relation types.

---

## 13.2 Notebook 02 — Knowledge Graph Construction

File:

```text
notebooks/02_knowledge_graph_construction.ipynb
```

### Purpose

This notebook converts the annotated dataset into a directed NetworkX Knowledge Graph.

It:

- creates graph nodes from annotated entity mentions;
- creates directed graph edges from annotated relations;
- checks source and target references;
- separates valid and invalid/malformed relation records;
- stores node and edge metadata;
- calculates basic graph counts;
- serialises the graph for later notebooks.

### Main input

```text
data/archive/all.jsonl
```

### Main outputs

```text
outputs/cti_knowledge_graph.pkl
outputs/kg_entities.csv
outputs/kg_relations_all.csv
outputs/kg_relations_valid.csv
outputs/kg_graph_summary.csv
outputs/kg_node_type_distribution.csv
outputs/kg_edge_type_distribution.csv
outputs/kg_node_edge_type_distribution.png
```

### Expected original graph

Approximately:

```text
Nodes:                         12,679
Directed edges:                1,094
Weakly connected components:  11,672
```

The large number of components relative to the number of nodes demonstrates severe graph fragmentation.

---

## 13.3 Notebook 03 — Graph Analytics

File:

```text
notebooks/03_graph_analytics.ipynb
```

### Purpose

This notebook analyses the structure of the original CTI graph before refinement.

It includes:

- node degree analysis;
- in-degree/out-degree interpretation;
- weakly connected component analysis;
- PageRank;
- betweenness centrality;
- combined importance scoring;
- community detection/summary;
- important-node inspection by entity type;
- graph-level structural diagnostics.

### Main input

```text
outputs/cti_knowledge_graph.pkl
```

### Main outputs

```text
outputs/graph_basic_statistics.csv
outputs/graph_degree_analysis.csv
outputs/graph_component_analysis.csv
outputs/graph_pagerank_scores.csv
outputs/graph_betweenness_scores.csv
outputs/graph_combined_importance_scores.csv
outputs/graph_important_entities_by_type.csv
outputs/community_summary.csv
outputs/community_entity_type_summary.csv
outputs/largest_community_relation_summary.csv
outputs/node_communities.csv
outputs/top_nodes_per_community.csv
outputs/degree_distribution.png
outputs/graph_relation_type_distribution.csv
```

### What success looks like

The notebook should confirm that the original graph is extremely sparse and fragmented. This observation motivates Notebook 04.

---

## 13.4 Notebook 04 — Entity Canonicalisation and Graph Refinement

File:

```text
notebooks/04_entity_canonicalisation_graph_refinement.ipynb
```

### Purpose

The same real-world CTI entity can appear under repeated or slightly different textual forms. This notebook reduces duplicate mentions so that structurally related information is less fragmented across many nodes.

The notebook evaluates progressively stronger refinement stages:

- **V1:** exact match after text normalisation;
- **V2:** noise filtering;
- **V3:** punctuation-insensitive refinement;
- **V4:** fuzzy matching, producing the final graph used downstream.

It also saves mapping tables and a manual-validation sample so that merges can be inspected.

### Main inputs

```text
outputs/cti_knowledge_graph.pkl
data/archive/all.jsonl
```

### Main outputs

```text
outputs/cti_knowledge_graph_canonical_v2.pkl
outputs/cti_knowledge_graph_canonical_v3.pkl
outputs/cti_knowledge_graph_canonical_final.pkl
outputs/canonical_entity_mapping_v2.csv
outputs/canonical_entity_mapping_v3.csv
outputs/canonical_entity_mapping_v4.csv
outputs/canonical_merge_examples.csv
outputs/canonicalisation_v4_manual_validation_sample.csv
outputs/kg_canonical_entities_v2.csv
outputs/kg_canonical_entities_v3.csv
outputs/kg_canonical_entities_v4.csv
outputs/kg_canonical_relations_v2.csv
outputs/kg_canonical_relations_v3.csv
outputs/kg_canonical_relations_v4.csv
outputs/graph_original_vs_canonical_v1_v2_v3_v4_comparison.csv
```

### Expected graph refinement summary

| Graph version | Nodes | Edges | Weakly connected components |
|---|---:|---:|---:|
| Original mention-level | 12,679 | 1,094 | 11,672 |
| V1 — exact normalised | 4,385 | 945 | 3,571 |
| V2 — noise filtered | 4,363 | 935 | 3,557 |
| V3 — punctuation-insensitive | 4,354 | 933 | 3,550 |
| V4 — final canonical graph | 4,050 | 910 | 3,267 |

### Interpretation note

Canonicalisation is an experimental graph-refinement stage. A reduction in node count is not automatically proof that every merge is semantically perfect. The final interpretation should therefore use the manual merge sample and downstream comparative results alongside the structural reduction.

---

## 13.5 Notebook 05 — Node2Vec Graph Embeddings

File:

```text
notebooks/05_graph_embeddings_node2vec.ipynb
```

### Purpose

Machine-learning models require numerical inputs. This notebook converts the final graph structure into one numerical vector per canonical node using Node2Vec.

The embeddings are structural: they represent graph position and neighbourhood patterns rather than the raw text of the entity name.

### Main input

```text
outputs/cti_knowledge_graph_canonical_final.pkl
```

### Final Node2Vec configuration

```text
Embedding dimension:   128
Walk length:           20
Context size:          10
Walks per node:        10
p:                     1.0
q:                     1.0
Batch size:            128
Learning rate:         0.01
Epochs:                50
Seed:                  42
```

### Main outputs

```text
outputs/node2vec_embeddings/node2vec_embeddings.npy
outputs/node2vec_embeddings/node2vec_embeddings_with_metadata.csv
outputs/node2vec_embeddings/node2vec_embedding_quality_summary.csv
outputs/node2vec_embeddings/node2vec_link_prediction_evaluation.csv
outputs/node2vec_embeddings/node2vec_loss_history.csv
outputs/node2vec_embeddings/node2vec_model_state_dict.pt
outputs/node2vec_embeddings/node2vec_nearest_neighbour_example.csv
outputs/node2vec_embeddings/node2vec_training_summary.csv
outputs/node2vec_embeddings/node_index_mapping.csv
outputs/node2vec_embeddings/node_metadata.csv
outputs/node2vec_embeddings/figures/node2vec_pca_visualisation.png
outputs/node2vec_embeddings/figures/node2vec_training_loss.png
outputs/node2vec_embeddings/figures/node2vec_tsne_visualisation.png
```

### Expected final embedding dimensions

```text
Nodes:        4,050
Dimensions:   128
```

### Important evaluation note

The notebook reports a post-training edge-separation diagnostic of approximately:

```text
ROC-AUC:             0.9934
Average Precision:   0.9929
```

This is an **internal embedding-quality diagnostic**, not an unbiased held-out link-prediction result, because the Node2Vec embeddings are learned from the complete graph.

Notebook 09 contains the separate leakage-controlled association-detection experiment.

---

## 13.6 Notebook 06 — Traditional Machine-Learning Baseline

File:

```text
notebooks/06_traditional_ml_baseline.ipynb
```

### Purpose

This notebook establishes non-GNN baselines for the main entity-type classification task.

The target is the cybersecurity entity type and the model input is the 128-dimensional Node2Vec representation created in Notebook 05.

Models include:

- Majority Class baseline;
- Logistic Regression;
- Random Forest.

### Main input

```text
outputs/node2vec_embeddings/node2vec_embeddings_with_metadata.csv
```

### Fixed node split

```text
Train:        2,835 nodes
Validation:     607 nodes
Test:           608 nodes
Total:        4,050 nodes
Classes:         20
```

Model selection is performed using **validation Macro-F1**. The saved split is reused by Notebook 07 so that the traditional ML and GraphSAGE results are compared on the same node population.

### Main output folder

```text
outputs/ml_baseline/
```

The notebook creates the following principal result files:

```text
outputs/ml_baseline/
|-- figures/
|   |-- best_ml_model_normalised_confusion_matrix.png
|   |-- best_ml_model_test_confusion_matrix.png
|   |-- entity_type_distribution_ml_baseline.png
|   `-- validation_model_comparison.png
|-- models/
|   |-- best_traditional_ml_model.joblib
|   |-- entity_type_label_encoder.joblib
|   |-- logistic_regression_pipeline.joblib
|   |-- majority_class_model.joblib
|   `-- random_forest_model.joblib
|-- best_model_connectivity_analysis.csv
|-- best_model_misclassification_summary.csv
|-- best_model_test_classification_report.csv
|-- best_model_test_predictions.csv
|-- embedding_connectivity_summary.csv
|-- entity_type_class_distribution.csv
|-- entity_type_label_mapping.csv
|-- logistic_regression_test_report.csv
|-- logistic_regression_validation_report.csv
|-- majority_class_validation_report.csv
|-- ml_experiment_configuration.json
|-- node_classification_split_assignments.csv
|-- random_forest_feature_importance.csv
|-- random_forest_test_report.csv
|-- random_forest_validation_report.csv
|-- split_class_distribution.csv
|-- test_model_results.csv
|-- test_row_indices.npy
|-- train_row_indices.npy
|-- validation_model_results.csv
`-- validation_row_indices.npy
```


### Reproduction checkpoint

Confirm that this file exists before starting Notebook 07:

```text
outputs/ml_baseline/node_classification_split_assignments.csv
```

---

## 13.7 Notebook 07 — GraphSAGE Node Classification

File:

```text
notebooks/07_graphsage_node_classification.ipynb
```

### Purpose

This notebook performs the main Graph Neural Network experiment.

It evaluates a direction-aware GraphSAGE classifier using:

- the final canonical CTI graph;
- 128-dimensional Node2Vec node features;
- the exact fixed Notebook 06 split;
- separate incoming and outgoing graph aggregation;
- LayerNorm;
- early stopping;
- validation-only hyperparameter selection;
- repeated evaluation across ten random seeds.

### Main inputs

```text
outputs/cti_knowledge_graph.pkl
outputs/cti_knowledge_graph_canonical_final.pkl
outputs/node2vec_embeddings/node2vec_embeddings_with_metadata.csv
outputs/ml_baseline/node_classification_split_assignments.csv
```

### Evaluation seeds

```text
11, 22, 33, 42, 55, 66, 77, 88, 99, 111
```

### Hyperparameter-tuning seeds

```text
22, 42, 77
```

### Selected GraphSAGE configuration

```text
Hidden dimension:          64
Dropout:                   0.3
Learning rate:             0.005
Weight decay:              0.0005
Maximum epochs:            250
Minimum epochs:            25
Early-stopping patience:   25
Normalisation:             LayerNorm
Direction handling:        separate incoming/outgoing aggregation
Primary metric:            Macro-F1
```

### Connectivity diagnostic

The final canonical graph contains:

```text
Canonical nodes:          4,050
Connected nodes:            903
Isolated nodes:           3,147
Test nodes:                 608
Connected test nodes:       132
Isolated test nodes:        476
Isolated test percentage:  78.29%
```

Because most nodes are isolated, the notebook reports both:

1. the **complete canonical graph**, which remains the main system-level result; and
2. a **connected-node sensitivity analysis**, which measures GraphSAGE where graph-neighbour information is actually available.

### Main output folder

```text
outputs/graphsage_final/
```

Important outputs include:

```text
canonical_edge_diagnostics.csv
connectivity_validation.csv
counterfactual_edge_retention_diagnostics.csv
connected_class_eligibility.csv
counterfactual_class_support.csv
graphsage_hyperparameter_tuning.csv
full_graph_graphsage_seed_results.csv
connected_graphsage_seed_results.csv
graphsage_summary_mean_std.csv
full_graph_baseline_seed_results.csv
connected_baseline_seed_results.csv
full_graph_model_comparison.csv
connected_model_comparison.csv
original_vs_canonical_seed_results.csv
original_vs_canonical_summary.csv
prediction_confidence_summary.csv
representative_test_predictions.csv
representative_error_summary.csv
final_experiment_configuration.json
```

Models are stored under:

```text
outputs/graphsage_final/models/
```

Figures are stored under:

```text
outputs/graphsage_final/figures/
```

Counterfactual Node2Vec cache files are stored under:

```text
outputs/graphsage_final/cache/
```

### Expected key node-classification results

#### Full canonical graph

| Model | Macro-F1 |
|---|---:|
| GraphSAGE | **0.1108 ± 0.0091** |
| Random Forest | 0.0520 ± 0.0053 |
| Logistic Regression | 0.0497 |
| Majority Class | 0.0098 |

#### Connected-node sensitivity analysis

| Model | Macro-F1 |
|---|---:|
| GraphSAGE | **0.3212 ± 0.0326** |
| Logistic Regression | 0.1428 |
| Random Forest | 0.1324 ± 0.0124 |
| Majority Class | 0.0257 |

The connected-node result must not be reported as the overall complete-graph score.

### Original-versus-canonical counterfactual

| Condition | Macro-F1 |
|---|---:|
| Original graph | **0.1792 ± 0.0147** |
| Canonical graph | **0.3248 ± 0.0335** |

Absolute Macro-F1 gain:

```text
+0.1457
```

Relative gain:

```text
+81.31%
```

Because canonicalisation changes node identity and graph size, this is interpreted as a **pipeline-level counterfactual** rather than a perfectly matched-node causal experiment.

---

## 13.8 Notebook 08 — Final Comparative Evaluation

File:

```text
notebooks/08_final_comparative_evaluation.ipynb
```

### Purpose

This notebook consolidates the final experimental evidence into dissertation-ready tables and figures.

It:

- loads the final outputs from previous stages;
- summarises graph refinement;
- compares GraphSAGE with traditional baselines;
- compares full-graph and connected-node performance;
- compares original and canonical graph conditions;
- performs paired statistical tests;
- calculates effect sizes;
- summarises model confidence and major error pairs;
- saves final tables, figures and a machine-readable research summary.

### Main output folder

```text
outputs/final_evaluation/
```

Final tables are stored under:

```text
outputs/final_evaluation/tables/
```

Important tables include:

```text
final_full_graph_model_comparison.csv
final_connected_model_comparison.csv
final_original_vs_canonical_comparison.csv
final_graphsage_full_vs_connected.csv
final_paired_statistical_tests.csv
final_effect_size_summary.csv
final_connectivity_summary.csv
final_top_error_pairs.csv
final_experimental_protocol.csv
strengths_limitations_future_work.csv
```

Final figures are stored under:

```text
outputs/final_evaluation/figures/
```

Important figures include:

```text
full_graph_model_macro_f1.png
connected_model_macro_f1.png
graphsage_full_vs_connected_macro_f1.png
original_vs_canonical_macro_f1.png
```

Machine-readable summary:

```text
outputs/final_evaluation/final_research_summary.json
```

Human-readable final answers to RQ1 and RQ2:

```text
outputs/final_evaluation/final_research_answer.txt
```

### Expected paired statistical comparisons

| Comparison | Mean Macro-F1 difference | Wilcoxon p-value |
|---|---:|---:|
| GraphSAGE vs Random Forest — full graph | +0.0588 | 0.0020 |
| GraphSAGE vs Random Forest — connected | +0.1887 | 0.0020 |
| Canonical graph vs original graph | +0.1457 | 0.0020 |

The repeated-seed results show consistent improvements under these experimental conditions.

Notebook 08 also writes a dissertation-level answer for both research questions. The saved answer records:

- **RQ1:** original graph `0.1792 ± 0.0147` versus canonical graph `0.3248 ± 0.0335`, an absolute Macro-F1 gain of `0.1457` and relative gain of `81.31%`;
- **RQ2:** GraphSAGE `0.1108 ± 0.0091` on the full graph versus Random Forest `0.0520 ± 0.0053` and Logistic Regression `0.0497`, with GraphSAGE rising to `0.3212 ± 0.0326` on the connected-node sensitivity condition;
- paired Wilcoxon comparisons reported at `p = 0.0020`;
- the important limitation that approximately `78.29%` of the complete test population is isolated.

---

## 13.9 Notebook 09 — Threat Association Detection

File:

```text
notebooks/09_threat_association_detection_FINAL.ipynb
```

### Purpose

This is a secondary experiment that asks a different question from node classification:

> Given a source CTI entity and a target CTI entity, is a directed threat association likely to exist between them?

It compares:

- Logistic Regression;
- Random Forest;
- Directed GraphSAGE.

### Main graph input

```text
outputs/cti_knowledge_graph_canonical_final.pkl
```

Final graph:

```text
Nodes:  4,050
Edges:    910
Type:   directed
```

### Edge split

Positive directed edges are split approximately as:

```text
Train:        638
Validation:   136
Test:         136
```

Balanced type-compatible negative examples are generated for each split.

### Leakage controls

The experiment is designed so that:

- validation and test positive edges are excluded from GraphSAGE message passing;
- degree features are calculated from the training graph;
- the full-graph Node2Vec embeddings from Notebook 05 are not reused for this edge-prediction experiment;
- the decision threshold is selected using validation data rather than the test set.

### Node features used in Notebook 09

- one-hot entity type;
- `log1p` mention count;
- training-graph in-degree;
- training-graph out-degree.

### Main output folder

```text
outputs/threat_association_detection/
```

Subfolders include:

```text
outputs/threat_association_detection/figures/
outputs/threat_association_detection/models/
outputs/threat_association_detection/splits/
```

Important outputs include:

```text
baseline_detection_seed_results.csv
graphsage_detection_seed_results.csv
all_detection_seed_results.csv
detection_summary_mean_std.csv
detection_statistical_comparison.csv
representative_detection_test_predictions.csv
representative_false_positives.csv
representative_false_negatives.csv
top_candidate_threat_associations.csv
representative_training_history.csv
detection_experiment_configuration.json
```

A representative trained model is stored under:

```text
outputs/threat_association_detection/models/
```

### Expected mean results across ten seeds

| Model | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|
| Logistic Regression | **0.7426** | 0.8133 | **0.8363** |
| Random Forest | **0.7765** | **0.8349** | 0.8224 |
| Directed GraphSAGE | 0.3974 | 0.8032 | 0.8001 |

The result demonstrates that the GNN does not universally outperform simpler models. In this secondary association task, traditional pairwise models produce stronger overall F1/AP performance, while Directed GraphSAGE produces high precision but substantially lower recall.

### Candidate associations

`top_candidate_threat_associations.csv` contains model-scored **candidate** unobserved associations.

These are **not automatically confirmed cyber-threat facts**. They are intended only as candidates for analyst review or future external validation.

---

# 14. How to compare a reproduced run with the submitted run

A successful reproduction does not require every floating-point number to be character-for-character identical across every computer. Small differences can occur because of:

- operating-system differences;
- CPU implementation differences;
- package patch versions;
- stochastic optimisation;
- multithreading.

The important checks are:

1. the dataset loads correctly;
2. the original graph is approximately 12,679 nodes and 1,094 edges;
3. the final canonical graph is approximately 4,050 nodes and 910 edges;
4. Node2Vec produces 128 features per canonical node;
5. the Notebook 06 split contains 2,835 train, 607 validation and 608 test nodes;
6. Notebook 07 uses ten evaluation seeds;
7. GraphSAGE full-graph Macro-F1 is close to 0.111;
8. connected-node GraphSAGE Macro-F1 is close to 0.321;
9. the canonical counterfactual performs substantially better than the original counterfactual;
10. Notebook 09 produces average-precision values close to the reported range.

If the graph node/edge counts differ substantially, do not continue silently. Check the data files, notebook execution order and earlier output files first.

---

# 15. Main research results at a glance

## 15.1 Graph refinement

```text
Original graph:   12,679 nodes / 1,094 edges / 11,672 weak components
Final V4 graph:    4,050 nodes /   910 edges /  3,267 weak components
```

The refinement process substantially reduces duplicate-node fragmentation.

## 15.2 Full-graph node classification

```text
GraphSAGE Macro-F1:           0.1108 ± 0.0091
Random Forest Macro-F1:       0.0520 ± 0.0053
Logistic Regression Macro-F1: 0.0497
Majority Macro-F1:            0.0098
```

## 15.3 Connected-node sensitivity analysis

```text
GraphSAGE Macro-F1:           0.3212 ± 0.0326
Logistic Regression Macro-F1: 0.1428
Random Forest Macro-F1:       0.1324 ± 0.0124
Majority Macro-F1:            0.0257
```

## 15.4 Original versus canonical graph

```text
Original graph counterfactual Macro-F1:   0.1792 ± 0.0147
Canonical graph counterfactual Macro-F1:  0.3248 ± 0.0335
Absolute gain:                            0.1457
Relative gain:                            81.31%
```

## 15.5 Threat association detection

```text
Logistic Regression Average Precision:  0.8363
Random Forest Average Precision:        0.8224
Directed GraphSAGE Average Precision:   0.8001
```

---

# 16. Correct interpretation of the results

The following interpretation is important when reproducing or citing this work.

## 16.1 Full graph versus connected nodes

The complete canonical graph is the primary system-level evaluation.

GraphSAGE achieved approximately:

```text
0.111 Macro-F1 on the complete canonical test graph
```

and approximately:

```text
0.321 Macro-F1 on the connected-node sensitivity subset
```

The connected-node result should **not** be presented as the overall system score. It demonstrates how performance changes when graph neighbourhood information is available.

## 16.2 Why the difference is large

The final graph remains extremely sparse:

```text
4,050 total canonical nodes
903 connected nodes
3,147 isolated nodes
```

In the test split:

```text
608 total test nodes
132 connected test nodes
476 isolated test nodes
```

Therefore, approximately 78.29% of the test nodes have no graph neighbours from which GraphSAGE can aggregate information.

## 16.3 Canonical graph comparison

The original-versus-canonical experiment demonstrates improved downstream learnability under the refined graph condition.

It does not prove that every individual entity merge is semantically correct, because canonicalisation changes the graph's node population and identity structure.

## 16.4 Node2Vec link diagnostic

The very high Notebook 05 edge-separation score is an embedding-quality diagnostic only. It is not an unbiased held-out link-prediction claim.

## 16.5 Threat-association candidates

Predicted unobserved associations in Notebook 09 are candidates for analyst review, not verified intelligence.

---

# 17. Reproducibility settings

## Node2Vec

```text
Seed:             42
Dimensions:       128
Walk length:      20
Context size:     10
Walks per node:   10
p:                1.0
q:                1.0
Epochs:           50
Learning rate:    0.01
```

## Node-classification split

```text
Train:        2,835
Validation:     607
Test:           608
```

## GraphSAGE evaluation seeds

```text
11, 22, 33, 42, 55, 66, 77, 88, 99, 111
```

## GraphSAGE tuning seeds

```text
22, 42, 77
```

## Primary node-classification metric

```text
Macro-F1
```

## Primary association-detection metric

```text
Average Precision
```

Configuration JSON files and seed-level CSV files are saved by the later notebooks so that the exact submitted experiment settings can be inspected programmatically.

---

# 18. Common problems and fixes

## 18.1 `FileNotFoundError`

### Most common cause

JupyterLab was started from the wrong folder.

### Fix

Close JupyterLab, activate the project environment, move into:

```text
<PROJECT_ROOT>/notebooks/
```

and start it again:

```bash
python -m jupyter lab
```

Also check that the required dataset or previous-stage output file actually exists.

---

## 18.2 `ModuleNotFoundError`

### Most common cause

The notebook is using the wrong Python kernel, or `requirements.txt` was installed into a different environment.

### Fix

Activate the project environment and verify:

```bash
python --version
python -m pip --version
```

Then reinstall if necessary:

```bash
python -m pip install -r requirements.txt
```

In JupyterLab, select:

```text
Python 3.11 (CTI Graph Project)
```

---

## 18.3 `No module named pyg_lib`

Notebook 05's PyTorch Geometric Node2Vec implementation requires `pyg_lib`.

`pyg_lib` is intentionally installed separately from `requirements.txt`. With the submitted PyTorch 2.12.x CPU environment, run:

```bash
python -m pip install pyg_lib -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
```

Then verify:

```bash
python -c "import pyg_lib; print('pyg_lib: OK')"
```

---

## 18.4 PyTorch or PyTorch Geometric version conflict

Use a fresh Python 3.11 environment rather than repeatedly upgrading and downgrading packages inside an old environment.

The submitted graph-learning configuration used:

```text
PyTorch 2.12.1+cpu
PyTorch Geometric 2.8.0
```

Packages such as `torchvision` and `torchaudio` are not required by this project.

---

## 18.5 `TqdmWarning: IProgress not found`

This normally affects notebook progress-bar display rather than model correctness.

Update Jupyter widgets:

```bash
python -m pip install --upgrade jupyterlab ipywidgets
```

Restart JupyterLab afterwards.

---

## 18.6 Notebook 09 cannot find the graph

Check the `PROJECT_ROOT` configuration at the top of Notebook 09.

It must identify the project root — the folder containing:

```text
data/
notebooks/
outputs/
requirements.txt
```

Then confirm this file exists:

```text
outputs/cti_knowledge_graph_canonical_final.pkl
```

If it does not exist, Notebook 04 must be completed successfully first.

---

## 18.7 A later notebook reports a missing CSV/model

Do not manually create an empty replacement file.

Later notebooks depend on genuine outputs from earlier stages. Return to the notebook that should have generated the missing artifact, rerun it successfully, and then continue.

---

## 18.8 Results are slightly different

Small numerical variation can occur across machines. Check that:

- Python 3.11 is used;
- the same data files are present;
- notebooks were run 01 through 09 in order;
- the saved fixed split is reused;
- the reported seeds/configuration are unchanged;
- old or partially generated outputs are not being mixed with a new run.

Large differences in node count, edge count or split size indicate a setup/data problem rather than harmless floating-point variation.

---

# 19. Running through VS Code instead of JupyterLab

JupyterLab is recommended for the simplest reproduction because the notebook working-directory behaviour is clear.

If VS Code is used:

1. install the Python and Jupyter extensions;
2. open the extracted project;
3. select the Python interpreter belonging to the `cti-graph-ml` environment;
4. open notebooks in numerical order;
5. select the `Python 3.11 (CTI Graph Project)` kernel;
6. ensure the notebook working directory still allows relative access to `../data/archive` and `../outputs`;
7. run all cells from a clean kernel.

If relative-path errors occur in VS Code, use the JupyterLab procedure in Section 9.

---

# 20. Runtime expectations

Runtime depends on CPU speed, available memory and background processes.

General expectation:

- Notebooks 01-04 are primarily data/graph processing and should normally complete faster than the model-training stages.
- Notebook 05 trains Node2Vec and can take several minutes on CPU.
- Notebook 06 trains conventional classifiers and is usually comparatively quick.
- Notebook 07 performs hyperparameter evaluation and repeated GraphSAGE runs, so it may take longer than a single model training run.
- Notebook 08 mainly aggregates previously saved results and should be comparatively quick.
- Notebook 09 repeats association-detection models across multiple seeds and can also take several minutes.

Do not interrupt a training cell only because the notebook appears idle for a short period. Check the kernel indicator and terminal/Jupyter status first.

---

# 21. Technology stack

The project uses:

- **Python** — implementation language;
- **JupyterLab** — notebook execution environment;
- **pandas / NumPy** — data processing;
- **NetworkX** — directed Knowledge Graph construction and analysis;
- **Matplotlib** — figures;
- **scikit-learn** — traditional ML, metrics and preprocessing;
- **SciPy** — statistical testing/supporting scientific operations;
- **PyTorch** — neural-network training;
- **PyTorch Geometric** — Node2Vec and GraphSAGE graph-learning components;
- **joblib** — saved traditional ML models;
- **PCA and t-SNE** — embedding visualisation.

---
