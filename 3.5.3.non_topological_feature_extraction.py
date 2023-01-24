from py2neo import Graph
import pandas as pd
import numpy as np
#%% use the Bolt URL and Password of your Neo4j database instance to connect
graph = Graph("bolt://localhost:7687", auth=("neo4j", "Neo4j"))

#%% load co_exploitation dataset: node1,node2, and label
training_df=pd.read_pickle('./GD_VCBD_Raw/co_exploitation_training_df.pkl')
test_df=pd.read_pickle('./GD_VCBD_Raw/co_exploitation_test_df.pkl')

#%% Export all properties of nodes in train and test sets
# define a function to extract properties of Vulnerability nodes
def extract_Vulnerability_property(data):
   query = """
    UNWIND $pairs AS pair
    MATCH (n1:Vulnerability) WHERE id(n1) = pair.node1
    MATCH (n2:Vulnerability) WHERE id(n2) = pair.node2
    RETURN id(n1) AS node1,
          id(n2) AS node2, 
          n1.cveID AS n1cveID, 
          n1.description AS n1description,
          n1.publishedDate AS n1publishedDate,
          n1.descriptionEmbedding20 AS n1descriptionEmbedding20,
          n1.numOfReference AS n1numOfReference,
          n1.v2version AS n1v2version,
          n1.v2baseScore AS n1v2baseScore,
          n1.v2accessVector AS n1v2accessVector,
          n1.v2accessComplexity AS n1v2accessComplexity,
          n1.v2authentication AS n1v2authentication,
          n1.v2confidentialityImpact AS n1v2confidentialityImpact,
          n1.v2integrityImpact AS n1v2integrityImpact,
          n1.v2availabilityImpact AS n1v2availabilityImpact,
          n1.v2vectorString AS n1v2vectorString,
          n1.v2impactScore AS n1v2impactScore,
          n1.v2exploitabilityScore AS n1v2exploitabilityScore,
          n1.v2userInteractionRequired AS n1v2userInteractionRequired,
          n1.v2severity AS n1v2severity,
          n1.v2obtainUserPrivilege AS n1v2obtainUserPrivilege,
          n1.v2obtainAllPrivilege AS n1v2obtainAllPrivilege,
          n1.v2acInsufInfo AS n1v2acInsufInfo,
          n1.v2obtainOtherPrivilege AS n1v2obtainOtherPrivilege,
          n1.cweid AS n1cweid,
          n1.gainedAccess AS n1gainedAccess,
          n1.vulnerabilityType AS n1vulnerabilityType,
          n2.cveID AS n2cveID,
          n2.description AS n2description,
          n2.publishedDate AS n2publishedDate,
          n2.descriptionEmbedding20 AS n2descriptionEmbedding20,
          n2.numOfReference AS n2numOfReference,
          n2.v2version AS n2v2version,
          n2.v2baseScore AS n2v2baseScore,
          n2.v2accessVector AS n2v2accessVector,
          n2.v2accessComplexity AS n2v2accessComplexity,
          n2.v2authentication AS n2v2authentication,
          n2.v2confidentialityImpact AS n2v2confidentialityImpact,
          n2.v2integrityImpact AS n2v2integrityImpact,
          n2.v2availabilityImpact AS n2v2availabilityImpact,
          n2.v2vectorString AS n2v2vectorString,
          n2.v2impactScore AS n2v2impactScore,
          n2.v2exploitabilityScore AS n2v2exploitabilityScore,
          n2.v2userInteractionRequired AS n2v2userInteractionRequired,
          n2.v2severity AS n2v2severity,
          n2.v2obtainUserPrivilege AS n2v2obtainUserPrivilege,
          n2.v2obtainAllPrivilege AS n2v2obtainAllPrivilege,
          n2.v2acInsufInfo AS n2v2acInsufInfo,
          n2.v2obtainOtherPrivilege AS n2v2obtainOtherPrivilege,
          n2.gainedAccess AS n2gainedAccess,
          n2.cweid AS n2cweid,
          n2.vulnerabilityType AS n2vulnerabilityType
   """
   pairs = [{"node1": node1, "node2": node2} for node1, node2 in data[["node1", "node2"]].values.tolist()]
   features = graph.run(query, {"pairs": pairs}).to_data_frame()
   return features


# Export properties of nodes in the training set
training_non_topological_features=extract_Vulnerability_property(training_df)
training_nonTopoDataset_df=pd.concat([training_df['label'], training_non_topological_features],axis=1)
#convert from neotime.Date format to dataframe datetime format, otherwise error happens
training_nonTopoDataset_df['n1publishedDate'] = pd.to_datetime({'year': [i.year for i in training_nonTopoDataset_df['n1publishedDate']],
                                                                'month':[i.month for i in training_nonTopoDataset_df['n1publishedDate']],
                                                                'day':[i.day for i in training_nonTopoDataset_df['n1publishedDate']]})
training_nonTopoDataset_df['n2publishedDate'] = pd.to_datetime({'year': [i.year for i in training_nonTopoDataset_df['n2publishedDate']],
                                                                'month':[i.month for i in training_nonTopoDataset_df['n2publishedDate']],
                                                                'day':[i.day for i in training_nonTopoDataset_df['n2publishedDate']]})
print(training_nonTopoDataset_df.head(10))
print(training_nonTopoDataset_df.columns.to_list())

# Export properties of nodes in the test set
test_non_topological_features = extract_Vulnerability_property(test_df)
test_nonTopoDataset_df=pd.concat([test_df['label'], test_non_topological_features],axis=1)
#convert from neotime.Date format to dataframe datetime format, otherwise error happens
test_nonTopoDataset_df['n1publishedDate'] = pd.to_datetime({'year': [i.year for i in test_nonTopoDataset_df['n1publishedDate']],
                                                                'month':[i.month for i in test_nonTopoDataset_df['n1publishedDate']],
                                                                'day':[i.day for i in test_nonTopoDataset_df['n1publishedDate']]})
test_nonTopoDataset_df['n2publishedDate'] = pd.to_datetime({'year': [i.year for i in test_nonTopoDataset_df['n2publishedDate']],
                                                            'month':[i.month for i in test_nonTopoDataset_df['n2publishedDate']],
                                                            'day':[i.day for i in test_nonTopoDataset_df['n2publishedDate']]})
print(test_nonTopoDataset_df.head(10))
print(test_nonTopoDataset_df.columns.to_list())

#%% save the constructed dataset in two formats: dataframe and csv
# save Xraw_train, ytrain, Xraw_test,ytest
# save in df format
training_nonTopoDataset_df.to_pickle('./GD_VCBD_Raw/GD_VCBD_Xraw_train_ytrain_df.pkl')
test_nonTopoDataset_df.to_pickle('./GD_VCBD_Raw/GD_VCBD_Xraw_test_ytest_df.pkl')
# save in csv format
training_nonTopoDataset_df.to_csv('./GD_VCBD_Raw/GD_VCBD_Xraw_train_ytrain.csv',index=False)
test_nonTopoDataset_df.to_csv('./GD_VCBD_Raw/GD_VCBD_Xraw_test_ytest.csv',index=False)

#%% Extract non-topological features from Xraw train and Xraw test
# denoted as Xnontopo_train and Xnontopo_test.
def nonTopoFeature_extraction(nonTopoDataset_df, dir=None):
    ###  1.unfold  n1descriptionEmbedding20  n2descriptionEmbedding20 seperately
    n1descriptionEmbedding20=np.array(
        [np.array(arrs) for arrs in nonTopoDataset_df['n1descriptionEmbedding20']])
    n2descriptionEmbedding20 = np.array(
        [np.array(arrs) for arrs in nonTopoDataset_df['n2descriptionEmbedding20']])

    label = np.array(nonTopoDataset_df['label']).reshape((-1, 1))
    node1 = np.array(nonTopoDataset_df['node1']).reshape((-1, 1))
    node2 = np.array(nonTopoDataset_df['node2']).reshape((-1, 1))

    ### 2. delete unnecessary columns
    delete_columns = ['n1publishedDate', 'n1descriptionEmbedding20','n1cweid','n1vulnerabilityType',
                      'n1cveID', 'n1description', 'n1v2version', 'n1v2vectorString',
                      'n2publishedDate', 'n2descriptionEmbedding20','n2cweid', 'n2vulnerabilityType',
                      'n2cveID','n2description','n2v2version','n2v2vectorString']
    nonTopoDataset_df = nonTopoDataset_df.drop(delete_columns, axis=1)

    ### 3. keep all numerial features
    n1numericalFeature4=np.array(nonTopoDataset_df[['n1numOfReference','n1v2baseScore',
                                                   'n1v2impactScore', 'n1v2exploitabilityScore'
                                                   ]]).reshape((-1, 4))
    n2numericalFeature4= np.array(nonTopoDataset_df[['n2numOfReference', 'n2v2baseScore',
                                                     'n2v2impactScore', 'n2v2exploitabilityScore'
                                                     ]]).reshape((-1, 4))

    ### 4. convert all boolean features to int
    nonTopoDataset_df[['n1v2obtainUserPrivilege', 'n1v2obtainAllPrivilege',
                       'n1v2userInteractionRequired','n1v2acInsufInfo',
                       'n1v2obtainOtherPrivilege',
                       'n2v2obtainUserPrivilege', 'n2v2obtainAllPrivilege',
                       'n2v2userInteractionRequired', 'n2v2acInsufInfo',
                       'n2v2obtainOtherPrivilege']]\
        =nonTopoDataset_df[['n1v2obtainUserPrivilege', 'n1v2obtainAllPrivilege',
                       'n1v2userInteractionRequired','n1v2acInsufInfo',
                       'n1v2obtainOtherPrivilege',
                       'n2v2obtainUserPrivilege', 'n2v2obtainAllPrivilege',
                       'n2v2userInteractionRequired', 'n2v2acInsufInfo',
                       'n2v2obtainOtherPrivilege']].fillna(2) ### 0,1,2

    nonTopoDataset_df[['n1v2obtainUserPrivilege', 'n1v2obtainAllPrivilege',
                       'n1v2userInteractionRequired','n1v2acInsufInfo',
                       'n1v2obtainOtherPrivilege',
                       'n2v2obtainUserPrivilege', 'n2v2obtainAllPrivilege',
                       'n2v2userInteractionRequired', 'n2v2acInsufInfo',
                       'n2v2obtainOtherPrivilege']]\
        = nonTopoDataset_df[['n1v2obtainUserPrivilege', 'n1v2obtainAllPrivilege',
                       'n1v2userInteractionRequired','n1v2acInsufInfo',
                       'n1v2obtainOtherPrivilege',
                       'n2v2obtainUserPrivilege', 'n2v2obtainAllPrivilege',
                       'n2v2userInteractionRequired', 'n2v2acInsufInfo',
                       'n2v2obtainOtherPrivilege']].astype(int)
    n1booleanFeature5=np.array(nonTopoDataset_df[['n1v2obtainUserPrivilege', 'n1v2obtainAllPrivilege',
                       'n1v2userInteractionRequired','n1v2acInsufInfo',
                       'n1v2obtainOtherPrivilege']]).reshape((-1, 5))
    n2booleanFeature5=np.array(nonTopoDataset_df[['n2v2obtainUserPrivilege', 'n2v2obtainAllPrivilege',
                       'n2v2userInteractionRequired', 'n2v2acInsufInfo',
                       'n2v2obtainOtherPrivilege']]).reshape((-1, 5))

    ### 5. convert category features to numerical features
    cate_columns=['n1v2accessVector','n1v2accessComplexity','n1v2authentication', 'n1v2confidentialityImpact',
                   'n1v2integrityImpact','n1v2availabilityImpact','n1v2severity', 'n1gainedAccess',
                   'n2v2accessVector', 'n2v2accessComplexity', 'n2v2authentication', 'n2v2confidentialityImpact',
                   'n2v2integrityImpact', 'n2v2availabilityImpact', 'n2v2severity', 'n2gainedAccess']
    nonTopoDataset_df[cate_columns] = nonTopoDataset_df[cate_columns].astype('category')
    for col in cate_columns:
        nonTopoDataset_df[col] = nonTopoDataset_df[col].cat.codes

    n1categoryFeature8=np.array(nonTopoDataset_df[['n1v2accessVector','n1v2accessComplexity','n1v2authentication', 'n1v2confidentialityImpact',
                   'n1v2integrityImpact','n1v2availabilityImpact','n1v2severity', 'n1gainedAccess']]).reshape((-1,8))
    n2categoryFeature8 = np.array(nonTopoDataset_df[['n2v2accessVector', 'n2v2accessComplexity', 'n2v2authentication', 'n2v2confidentialityImpact',
                   'n2v2integrityImpact', 'n2v2availabilityImpact', 'n2v2severity', 'n2gainedAccess']]).reshape((-1, 8))

    # set NaN to 0
    n1descriptionEmbedding20[np.isnan(n1descriptionEmbedding20)] = 0
    n1numericalFeature4[np.isnan(n1numericalFeature4)] = 0
    n1booleanFeature5[np.isnan(n1booleanFeature5)] = 0
    n1categoryFeature8[np.isnan(n1categoryFeature8)] = 0
    n2descriptionEmbedding20[np.isnan(n2descriptionEmbedding20)] = 0
    n2numericalFeature4[np.isnan(n2numericalFeature4)] = 0
    n2booleanFeature5[np.isnan(n2booleanFeature5)] = 0
    n2categoryFeature8[np.isnan(n2categoryFeature8)] = 0

    # concatenate all np array features
    nonTopoDatasetArray=np.concatenate((label,node1, node2, n1descriptionEmbedding20, n1numericalFeature4,
                                        n1booleanFeature5,n1categoryFeature8,
                                        n2descriptionEmbedding20,n2numericalFeature4,
                                        n2booleanFeature5,n2categoryFeature8), axis=1)
    nonTopoDatasetDict={
        'label':label,
        'node1':node1,
        'node2':node2,
        'n1descriptionEmbedding20':n1descriptionEmbedding20,
        'n1numericalFeature4':n1numericalFeature4,
        'n1booleanFeature5':n1booleanFeature5,
        'n1categoryFeature8':n1categoryFeature8,
        'n2descriptionEmbedding20': n2descriptionEmbedding20,
        'n2numericalFeature4': n2numericalFeature4,
        'n2booleanFeature5': n2booleanFeature5,
        'n2categoryFeature8':n2categoryFeature8
    }

    np.savez(dir, nonTopoDatasetArray=nonTopoDatasetArray,
             nonTopoDatasetDict=nonTopoDatasetDict)
    return nonTopoDatasetArray,nonTopoDatasetDict

#% get nonTopoDataset in Arrays
nonTopoFeature_extraction(training_nonTopoDataset_df,
                           dir='./GD_VCBD_Raw/GD_VCBD_go_Xnontopo_train')
nonTopoFeature_extraction(test_nonTopoDataset_df,
                          dir='./GD_VCBD_Raw/GD_VCBD_go_Xnontopo_test')


