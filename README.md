# VulDG
This repo contains the data and codes for the paper submitted to IJCAI 2023, titled "A Comprehensive Graph Dataset for Software Vulnerability Assessment".

## 1. Repo Structure


```
VulGD
├── README.md 
├── import
│   ├── AffectsAddProperty.csv
│   ├── DomainNodes_Vulnerabiliy_HAS_REFERENCE_Domain_relationship.csv
│   ├── ExploitNodes.csv
│   ├── ProductNodes_VendorNodes_Vulnerability_AFFECTS_Product_BELONGS_TO_Vendor.csv
│   ├── VulnerabilityNodes.csv
│   ├── VulnerabilityNodesAddProperties.csv
│   ├── Vulnerabiliy_HAS_EXPLOIT_Exploit_relationship.csv
│   └── WeaknessNodes.csv
├── DescriptionEmbedding
│   └── VulnerabilityNodesTextEmbedding20.pkl
├── 2.VulGD_Deployment_Cypher.cypher
├── 3.3.VCBD_link_prediction.py
├── 3.4.1.Plot_F1_score_comparison.py
├── 3.5.1.select_co_exploitation_links_in_training_and_test_sets.py
├── 3.5.2.CO_AFFECT_subgraph_and_topological_feature_extraction.py
├── 3.5.3.non_topological_feature_extraction.py
└── 3.5.3.non_topological_feature_extraction.py
```

Folder **import** contains all original data for VulGD deployment.

Folder **DescriptionEmbedding** contains the 20-dimensional extracted feature from vulnerability descriptions using a pre-trained BERT model. This file will be used in **3.5.3.non_topological_feature_extraction.py**. 

File **2.VulGD_Deployment_Cypher.cypher** contains the cypher codes for VulGD deployment on the Neo4j graph database platform, described in Section 2. 

Files 3.3 - 3.5.3 are the python codes for the use case, Vulnerability Co-Exploitation Behaviour Discovery (VCBD), on the VulGD.



## 2. VulGD Deployment
This section introduces how to deploy the VulGD into the Neo4j graph database platform.
### 2.1 Programming Language
[Cypher Query Language](https://neo4j.com/developer/cypher/)
### 2.2 Software/Platform
Neo4j Desktop
### 2.3 A step-by-step guide for VulGD Deployment
1. Download [(from here)](https://neo4j.com/download/) and install [(refer here)]( https://neo4j.com/docs/desktop-manual/current/installation/) Neo4j Desktop 1.4.15 or higher versions
2. Create a project named **VulGD Project** with Neo4j Desktop.
3. Add a local DBMS named **Graph DBMS** with Neo4j Desktop and set the password as **Neo4j**. Choose version 4.4.11 for **Graph DBMS**.
4. Start **Graph DBMS**
5. Install APOC [(refer here)](https://neo4j.com/labs/apoc/4.3/installation/) and Graph Data Science Library [(refer here)](https://neo4j.com/docs/graph-data-science/current/installation/neo4j-desktop/) plugins for **Graph DBMS** with Neo4j Desktop.
 
6. Open the setting file of **Graph DBMS** and add a line as below in the setting file. 
```
    apoc.import.file.enabled=true
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;to tackle an error:

```
    Failed to invoke procedure `apoc.periodic.iterate`: Caused by: 
    java.lang.RuntimeException: Import from files not enabled, please set 
    apoc.import.file.enabled=true in your apoc.conf
```
7. Put all files in the [import](import/) folder into the **import** folder of **Graph DBMS**.
8. Open **Graph DBMS** with **Neo4j Browser**. Since Neo4j Browser comes out-of-the-box when you install Neo4j Desktop on your system, no installation is required.
9.  Click the **Enable multi statement query editor** to enable running multiple Cypher statements separated by semi-colons **;** in the Neo4j Browser setting.
10. Run Cypher statements in the **2.VulGD_Deployment_Cypher.cypher** file with the Neo4j Browser to deploy VulKG.

## 3. Use case: Vulnerability Co-Exploitation Behaviour Discovery
This section introduces how to implement the use case: Vulnerability Co-exploitation Behaviour Discovery (VCBD) on VulGD.

### 3.1 Programing language
Python and Cypher

### 3.2 Library
numpy==1.22.4

scikit-learn==1.1.1

matplotlib==3.5.2

### 3.3 Data
Data is provided in folder **GD_VCBD_Ready_to_go**. The generation process of this subgraph dataset is described in Section 3.5. 

### 3.3 Vulnerability Co-Exploitation Behaviour Discovery

Run python codes in 
**3.3.VCBD_link_prediction.py** to get the results reported in Table 7 and Table 8. 

### 3.4 Result Visualization

Run python codes in 
**3.4.1.Plot_F1_score_comparison.py** to get  the visualization results reported in Fig. 4.

Run python codes in **3.4.2.Plot_ROC.py** to get the visualization results reported in Fig. 5.

### 3.5 Subgraph dataset generation for VCBD task

This subsection introduces how to generate a raw version and a ready-to-go version of graph datasets for the VCBD task, which are provided in folders named **GD_VCBD_Raw** and **GD_VCBD_Ready_to_go**.  In case someone wants to know the details on how to extract subgraph datasets from VulGD.  

#### 3.5.1 Programing language
Python and Cypher

#### 3.5.2 Library
py2neo==2021.2.3

pandas==1.4.2

numpy==1.22.4

#### 3.5.3 A step-by-step guide

1. Open Neo4j Desktop and start Graph DBMS
2. Open the setting file of Graph DBMS. Search and change the memory setting as below 
   ``` 
    dbms.memory.heap.initial_size=4G
    dbms.memory.heap.max_size=4G
   ```
to tackle an error:
```
    py2neo.errors.ClientError: [Procedure.ProcedureCallFailed] Failed to invoke procedure 
    `gds.graph.project.cypher`: Caused by: java.lang.OutOfMemoryError: Java heap space
```
4. run python codes in 
**3.5.1.select_co_exploitation_links_in_training_and_test_sets.py** to construct the link head-tail pairs in the training and test sets
5. run python codes in 
**3.5.2.CO_AFFECT_subgraph_and_topological_feature_extraction.py**  to extract the CO_AFFECT subgraph and topological features.
6. run python codes in 
**3.3.non_topological_feature_extraction.py** to extract non-topological features.

Once done, the generated GD-VCBD subgraph datasets will be saved in the corresponding folders, **GD_VCBD_Raw** and **GD_VCBD_Ready_to_go**.

# The end!
