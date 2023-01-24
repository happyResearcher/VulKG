from py2neo import Graph
import pandas as pd
import numpy as np

#%% use the Bolt URL and Password of your Neo4j database instance to connect
graph = Graph("bolt://localhost:7687", auth=("neo4j", "Neo4j"))

#%% load co_exploitation dataset: node1,node2, and label
training_df=pd.read_pickle('./GD_VCBD_Raw/co_exploitation_training_df.pkl')
test_df=pd.read_pickle('./GD_VCBD_Raw/co_exploitation_test_df.pkl')

#%% Extract the co-affect training sub-graph
# SET Vulnerability property: n.coExploitationTrainingNode=1 and then create a named graph based on this property
#Training set: only consider the training set link related nodes
trainingIDlist=list(set(training_df["node1"].to_list()+training_df["node2"].to_list()))
query = """
 MATCH (n:Vulnerability) WHERE id(n) in $IDlist
 SET n.coExploitationTrainingNode=1  //or null
 RETURN count(*)
"""
res = graph.run(query,{"IDlist": trainingIDlist}).to_data_frame()
print(res)

# Extract the co-affect training sub-graph
# leverage the (n1)-[:AFFECTS]->(p:Product)<-[:AFFECTS]-(n2) path information
query="""
CALL gds.graph.project.cypher(
'Co_Exploitation_Co_Affect_Training',
'MATCH (n:Vulnerability)
 WHERE n.coExploitationTrainingNode=1
 RETURN id(n) AS id',
 'MATCH (n1)-[:AFFECTS]->(p:Product)<-[:AFFECTS]-(n2) // only match rel on defined nodes
 WHERE n1.coExploitationTrainingNode=1 and n2.coExploitationTrainingNode=1 and id(n1)<id(n2)
 RETURN id(n1) AS source, id(n2) AS target, "CO_AFFECT" AS type, count(p) AS weight'
)
"""
res=graph.run(query).to_data_frame()
print("nodeCount:",res.nodeCount)
print("relationshipCount:",res.relationshipCount)

#%% Extract the co-affect test sub-graph
#% test set: : only consider the test set link related nodes
testIDlist=list(set(test_df["node1"].to_list()+test_df["node2"].to_list()))
query = """
 MATCH (n:Vulnerability) WHERE id(n) in $IDlist
 SET n.coExploitationTestNode=1  //or null
 RETURN count(*)
"""
res = graph.run(query,{"IDlist": testIDlist}).to_data_frame()
print(res)

# Extract the co-affect test sub-graph
# leverage the (n1)-[:AFFECTS]->(p:Product)<-[:AFFECTS]-(n2) path information
query="""
CALL gds.graph.project.cypher(
'Co_Exploitation_Co_Affect_Test',
'MATCH (n:Vulnerability)
 WHERE n.coExploitationTestNode=1
 RETURN id(n) AS id',
 'MATCH (n1)-[:AFFECTS]->(p:Product)<-[:AFFECTS]-(n2)
 WHERE n1.coExploitationTestNode=1 and n2.coExploitationTestNode=1 and id(n1)<id(n2)
 RETURN id(n1) AS source, id(n2) AS target, "CO_AFFECT" AS type, count(p) AS weight' 
)
"""

res=graph.run(query).to_data_frame()
print("nodeCount:",res.nodeCount)
print("relationshipCount:",res.relationshipCount)

# check the existance of all subgraphs
query = """CALL gds.graph.list"""
res = graph.run(query).to_data_frame()
print(res.graphName)

#%% Export the CO_AFFECT relationships in the format of (head, tail) pairs
# training set
query = """
CALL gds.beta.graph.relationships.stream(
"Co_Exploitation_Co_Affect_Training"
)
YIELD
    sourceNodeId,targetNodeId,relationshipType
RETURN
    sourceNodeId as head, targetNodeId as tail
ORDER BY head ASC, tail ASC
"""
coaffect_relationship_train = graph.run(query).to_data_frame()

# test set
query = """
CALL gds.beta.graph.relationships.stream(
"Co_Exploitation_Co_Affect_Test"
)
YIELD
    sourceNodeId,targetNodeId,relationshipType
RETURN
    sourceNodeId as head, targetNodeId as tail
ORDER BY head ASC, tail ASC
"""
coaffect_relationship_test = graph.run(query).to_data_frame()

#%% save the constructed subgraphs in two formats: dataframe and csv
#  save R_train^(CA), R_test^(CA)
# save in df format
coaffect_relationship_train.to_pickle('./GD_VCBD_Ready_to_go/GD_VCBD_R_train_CA_df.pkl')
coaffect_relationship_test.to_pickle('./GD_VCBD_Ready_to_go/GD_VCBD_R_test_CA_df.pkl')

# save in csv format
coaffect_relationship_train.to_csv('./GD_VCBD_Ready_to_go/GD_VCBD_R_train_CA.csv',index=False)
coaffect_relationship_test.to_csv('./GD_VCBD_Ready_to_go/GD_VCBD_R_test_CA.csv',index=False)

#%% Extract topological features from co-affect sub-graph
#%% 1.Centrality algorithms: https://neo4j.com/docs/graph-data-science/current/algorithms/centrality/
# 1.1 The PageRank algorithm:
# training set
res=graph.run("""
CALL gds.pageRank.write(
  'Co_Exploitation_Co_Affect_Training',
  {
  relationshipWeightProperty: 'weight',
  writeProperty:'pageRankTrain',
  maxIterations: 40,
  dampingFactor: 0.85
  }
  )
  YIELD nodePropertiesWritten, ranIterations, didConverge, centralityDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# test set
res=graph.run("""
CALL gds.pageRank.write(
  'Co_Exploitation_Co_Affect_Test',
  {
  relationshipWeightProperty: 'weight',
  writeProperty:'pageRankTest',
  maxIterations: 40,
  dampingFactor: 0.85
  }
  )
  YIELD nodePropertiesWritten, ranIterations, didConverge, centralityDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# 1.2 The Article Rank algorithm:
# training set
res=graph.run("""
CALL gds.articleRank.write(
  'Co_Exploitation_Co_Affect_Training',
  {
  relationshipWeightProperty: 'weight',
  writeProperty:'articleRankTrain',
  maxIterations: 100,
  dampingFactor: 0.15
  }
  )
  YIELD nodePropertiesWritten, ranIterations, didConverge, centralityDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# test set
res=graph.run("""
CALL gds.articleRank.write(
  'Co_Exploitation_Co_Affect_Test',
  {
  relationshipWeightProperty: 'weight',
  writeProperty:'articleRankTest',
  maxIterations: 100,
  dampingFactor: 0.15
  }
  )
  YIELD nodePropertiesWritten, ranIterations, didConverge, centralityDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# 1.3 The Degree Centrality
# training set
res=graph.run("""
CALL gds.degree.write(
  'Co_Exploitation_Co_Affect_Training',
  {
  writeProperty:'degreeTrain',
  relationshipWeightProperty: 'weight'
  }
  )
  YIELD nodePropertiesWritten, centralityDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# test set
res=graph.run("""
CALL gds.degree.write(
  'Co_Exploitation_Co_Affect_Test',
  {
  writeProperty:'degreeTest',
  relationshipWeightProperty: 'weight'
  }
  )
  YIELD nodePropertiesWritten, centralityDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

#% 1.4 Harmonic Centrality
# train graph
res=graph.run("""
CALL gds.alpha.closeness.harmonic.write(
  'Co_Exploitation_Co_Affect_Training',
  {
  writeProperty:'harmonicClosenessTrain'
  }
  )
  YIELD nodes, centralityDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# test set
res=graph.run("""
CALL gds.alpha.closeness.harmonic.write(
  'Co_Exploitation_Co_Affect_Test',
  {
  writeProperty:'harmonicClosenessTest'
  }
  )
  YIELD nodes, centralityDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])


# add centrality features to the train and test DataFrames:
def apply_centrality_features(data, feature_list):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.label AS lable,
    pair.node1 AS node1,
    pair.node2 AS node2,
    p1[$feature_list[0]] AS n1pageRank,
    p1[$feature_list[1]] AS n1articleRank,
    p1[$feature_list[2]] AS n1degree,
    p1[$feature_list[3]] AS n1harmonicCloseness,
    p2[$feature_list[0]] AS n2pageRank,
    p2[$feature_list[1]] AS n2articleRank,
    p2[$feature_list[2]] AS n2degree,
    p2[$feature_list[3]] AS n2harmonicCloseness
    """
    pairs = [{"node1": node1, "node2": node2,"label":label}  for node1,node2,label in data[["node1", "node2","label"]].values.tolist()]
    params = {
    "pairs": pairs,
    "feature_list": feature_list
    }
    features = graph.run(query, params).to_data_frame()
    return features

#%  Add the new centrality features:
feature_list_train=['pageRankTrain','articleRankTrain','degreeTrain','harmonicClosenessTrain']
feature_list_test=['pageRankTest','articleRankTest','degreeTest','harmonicClosenessTest']
training_centrality_df = apply_centrality_features(training_df, feature_list_train)
test_centrality_df = apply_centrality_features(test_df, feature_list_test)

print(training_centrality_df.head())
print(test_centrality_df.head())

features_name=training_centrality_df.columns[3:].to_list()

X_train = np.array(training_centrality_df[features_name]).reshape(-1,len(features_name))
y_train = np.array(training_centrality_df['lable'])

X_test = np.array(test_centrality_df[features_name]).reshape(-1,len(features_name))
y_test = np.array(test_centrality_df['lable'])

dir='./GD_VCBD_Ready_to_go/GD_VCBD_go_CA_Xtopo_centra'
np.savez(dir, X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)

#%% 2.Community detection algorithms: https://neo4j.com/docs/graph-data-science/current/algorithms/community/
# 2.1 Louvain
# training set
res=graph.run("""
CALL gds.louvain.write(
  'Co_Exploitation_Co_Affect_Training',
  {
  relationshipWeightProperty: 'weight',
  writeProperty:'louvainTrain',
  maxLevels:10,
  maxIterations:20,
  includeIntermediateCommunities: false
  }
  )
  YIELD nodePropertiesWritten, communityCount, modularities, communityDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# test set
res=graph.run("""
CALL gds.louvain.write(
  'Co_Exploitation_Co_Affect_Test',
  {
  relationshipWeightProperty: 'weight',
  writeProperty:'louvainTest',
  maxLevels:10,
  maxIterations:20,
  includeIntermediateCommunities: false
  }
  )
  YIELD nodePropertiesWritten, communityCount, modularities, communityDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# 2.2 Label Propagation
# training set
res=graph.run("""
CALL gds.labelPropagation.write(
  'Co_Exploitation_Co_Affect_Training',
  {
  relationshipWeightProperty: 'weight',
  writeProperty:'labelPropagationTrain',
  maxIterations: 100
  }
  )
  YIELD nodePropertiesWritten, ranIterations, didConverge, communityCount,communityDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# test set
res=graph.run("""
CALL gds.labelPropagation.write(
  'Co_Exploitation_Co_Affect_Test',
  {
  relationshipWeightProperty: 'weight',
  writeProperty:'labelPropagationTest',
  maxIterations: 100
  }
  )
  YIELD nodePropertiesWritten, ranIterations, didConverge, communityCount,communityDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# 2.3 Weakly Connected Components:
# training set
res=graph.run("""
CALL gds.wcc.write(
  'Co_Exploitation_Co_Affect_Training',
  {
  writeProperty:'wccTrain',
  relationshipWeightProperty: 'weight'
  }
  )
  YIELD nodePropertiesWritten, componentCount,componentDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# test set
res=graph.run("""
CALL gds.wcc.write(
  'Co_Exploitation_Co_Affect_Test',
  {
  writeProperty:'wccTest',
  relationshipWeightProperty: 'weight'
  }
  )
  YIELD nodePropertiesWritten, componentCount,componentDistribution;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# 2.4 Modularity Optimization
# trainging set
res=graph.run("""
CALL gds.beta.modularityOptimization.write(
  'Co_Exploitation_Co_Affect_Training',
  {
  writeProperty:'modularityTrain',
  relationshipWeightProperty: 'weight',
  maxIterations:100
  }
  )
  YIELD nodes, communityCount, ranIterations, didConverge;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# test set
res=graph.run("""
CALL gds.beta.modularityOptimization.write(
  'Co_Exploitation_Co_Affect_Test',
  {
  writeProperty:'modularityTest',
  relationshipWeightProperty: 'weight',
  maxIterations:100
  }
  )
  YIELD nodes, communityCount, ranIterations, didConverge;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

#  add community features to the train and test DataFrames:
def apply_community_features(data, feature_list):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.label AS lable,
    pair.node1 AS node1,
    pair.node2 AS node2,
    p1[$feature_list[0]] AS n1louvain,
    p1[$feature_list[1]] AS n1labelPropagation,
    p1[$feature_list[2]] AS n1wcc,
    p1[$feature_list[4]] AS n1modularity,
    p2[$feature_list[0]] AS n2louvain,
    p2[$feature_list[1]] AS n2labelPropagation,
    p2[$feature_list[2]] AS n2wcc,
    p2[$feature_list[4]] AS n2modularity
    """
    pairs = [{"node1": node1, "node2": node2,"label":label}  for node1,node2,label in data[["node1", "node2","label"]].values.tolist()]
    params = {
    "pairs": pairs,
    "feature_list": feature_list
    }
    features = graph.run(query, params).to_data_frame()
    return features
#  Add the new community  features:
feature_list_train=['louvainTrain','labelPropagationTrain','wccTrain','modularityTrain']
feature_list_test=['louvainTest','labelPropagationTrain','wccTest','modularityTest']

training_community_df = apply_community_features(training_df, feature_list_train)
test_community_df = apply_community_features(test_df, feature_list_test)

print(training_community_df.head())
print(test_community_df.head())

#  tackle ValueError: Input X contains NaN. solution: replace nan
training_community_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_community_df.replace([np.inf, -np.inf], np.nan, inplace=True)
training_community_df=training_community_df.fillna(0)
test_community_df=test_community_df.fillna(0)

features_name=training_community_df.columns[3:].to_list()

X_train = np.array(training_community_df[features_name]).reshape(-1,len(features_name))
y_train = np.array(training_community_df['lable'])

X_test = np.array(test_community_df[features_name]).reshape(-1,len(features_name))
y_test = np.array(test_community_df['lable'])

dir='./GD_VCBD_Ready_to_go/GD_VCBD_go_CA_Xtopo_commun'
np.savez(dir, X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)

#%% 3. Node embeddings : https://neo4j.com/docs/graph-data-science/current/algorithms/fastrp/#algorithms-embeddings-fastrp
# 3.1 Fast Random Projection
res=graph.run("""
CALL gds.fastRP.write(
  'Co_Exploitation_Co_Affect_Training',
  {
  writeProperty:'fastRPEmbeddingTrain',
  relationshipWeightProperty: 'weight',
  embeddingDimension: 20,
  randomSeed: 1
  }
  )
  YIELD nodePropertiesWritten;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# test set :
res=graph.run("""
CALL gds.fastRP.write(
  'Co_Exploitation_Co_Affect_Test',
  {
  writeProperty:'fastRPEmbeddingTest',
  relationshipWeightProperty: 'weight',
  embeddingDimension: 20,
  randomSeed: 1
  }
  )
  YIELD nodePropertiesWritten;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])


# 3.2 Node2Vec
# trainging set
res=graph.run("""
CALL gds.beta.node2vec.write(
  'Co_Exploitation_Co_Affect_Training',
  {
  writeProperty:'node2vecTrain',
  embeddingDimension: 20
  }
  )
  YIELD nodePropertiesWritten;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])

# test set :
res=graph.run("""
CALL gds.beta.node2vec.write(
  'Co_Exploitation_Co_Affect_Test',
  {
  writeProperty:'node2vecTest',
  embeddingDimension: 20
  }
  )
  YIELD nodePropertiesWritten;
""").to_data_frame()
for col in res.columns:
    print(col,res[col][0])


# The following function will add these features to the train and test DataFrames:
def apply_nodeEmbedding_features(data, featurName):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.label AS lable,
    pair.node1 AS node1,
    pair.node2 AS node2,
    p1[$featurName] AS n1embedding,
    p2[$featurName] AS n2embedding
    """
    pairs = [{"node1": node1, "node2": node2, "label": label} for node1, node2, label in
             data[["node1", "node2", "label"]].values.tolist()]
    params = {
        "pairs": pairs,
        "featurName": featurName
    }
    features = graph.run(query, params).to_data_frame()
    return features

#% Now apply the function to the training DataFrame: fastRPEmbeddingTrain
training_fastRPEmbedding20_df=apply_nodeEmbedding_features(training_df,'fastRPEmbeddingTrain')
#% Do the same to the test DataFrame:
test_fastRPEmbedding20_df = apply_nodeEmbedding_features(test_df,'fastRPEmbeddingTest')

n1fastRPEmbedding20=np.array(
        [np.array(arrs) for arrs in training_fastRPEmbedding20_df['n1embedding']])
n2fastRPEmbedding20=np.array(
        [np.array(arrs) for arrs in training_fastRPEmbedding20_df['n2embedding']])

X_train = np.concatenate((n1fastRPEmbedding20,n2fastRPEmbedding20), axis=1) #10698*40
y_train = np.array(training_fastRPEmbedding20_df['lable'])

n1fastRPEmbedding20_test=np.array(
        [np.array(arrs) for arrs in test_fastRPEmbedding20_df['n1embedding']])
n2fastRPEmbedding20_test=np.array(
        [np.array(arrs) for arrs in test_fastRPEmbedding20_df['n2embedding']])

X_test = np.concatenate((n1fastRPEmbedding20_test,n2fastRPEmbedding20_test), axis=1)
y_test = np.array(test_fastRPEmbedding20_df['lable'])

dir='./GD_VCBD_Ready_to_go/GD_VCBD_go_CA_Xtopo_fastRP'
np.savez(dir, X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)


#% Now apply the function to the training DataFrame: node2vecTrain
training_node2vec20_df=apply_nodeEmbedding_features(training_df,'node2vecTrain')
#% Do the same to the test DataFrame:
test_node2vec20_df = apply_nodeEmbedding_features(test_df,'node2vecTest')

n1node2vec20=np.array(
        [np.array(arrs) for arrs in training_node2vec20_df['n1embedding']])
n2node2vec20=np.array(
        [np.array(arrs) for arrs in training_node2vec20_df['n2embedding']])

X_train = np.concatenate((n1node2vec20,n2node2vec20), axis=1)
y_train = np.array(training_node2vec20_df['lable'])

n1node2vec20_test=np.array(
        [np.array(arrs) for arrs in test_node2vec20_df['n1embedding']])
n2node2vec20_test=np.array(
        [np.array(arrs) for arrs in test_node2vec20_df['n2embedding']])


X_test = np.concatenate((n1node2vec20_test,n2node2vec20_test), axis=1)
y_test = np.array(test_node2vec20_df['lable'])

dir='./GD_VCBD_Ready_to_go/GD_VCBD_go_CA_Xtopo_node2vec'
np.savez(dir, X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
