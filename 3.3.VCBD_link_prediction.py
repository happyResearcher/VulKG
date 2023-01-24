from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import numpy as np
import csv
from tqdm import trange

#%% creat empty list for unified workflow :
L_non=[] #temp list for storing data for non-topo feature
L_topo=[] #temp list for storing data for non-topo feature
L=[] #list for storing data for different combination of feature 

hyper_non=[] #temp list for storing hyper_setting for non-topo feature
hyper_topo=[] #temp list for storing hyper_setting for topo feature
hyper_list=[] #list for combined hyper_setting for all different combination of feature

dic_draw= {}
#%% load nonTopoDataset and creat train and test data for all-nontopo features:
dir='./GD_VCBD_Ready_to_go/GD_VCBD_go_Xnontopo_train'+'.npz'
data = np.load(dir,allow_pickle=True)
training_nonTopoDatasetArray=data['nonTopoDatasetArray']
X_train_allnon= training_nonTopoDatasetArray[:,3:]
y_train= training_nonTopoDatasetArray[:,0]
training_nonTopoDatasetDict=data['nonTopoDatasetDict'].item() # for dictionary

dir='./GD_VCBD_Ready_to_go/GD_VCBD_go_Xnontopo_test'+'.npz'
data = np.load(dir,allow_pickle=True)
test_nonTopoDatasetArray=data['nonTopoDatasetArray']
X_test_allnon= test_nonTopoDatasetArray[:,3:]
y_test= test_nonTopoDatasetArray[:,0]
test_nonTopoDatasetDict=data['nonTopoDatasetDict'].item() # for dictionary

L_non.append([X_train_allnon,y_train,X_test_allnon,y_test])
hyper_non.append('Nontopo_all')

#%% prepare data for description only:
n1description=training_nonTopoDatasetDict['n1descriptionEmbedding20']
n2description=training_nonTopoDatasetDict['n2descriptionEmbedding20']
X_train_descr = np.concatenate((n1description,n2description), axis=1)

n1description=test_nonTopoDatasetDict['n1descriptionEmbedding20']
n2description=test_nonTopoDatasetDict['n2descriptionEmbedding20']
X_test_descr = np.concatenate((n1description,n2description), axis=1)

L_non.append([X_train_descr,y_train,X_test_descr,y_test])
hyper_non.append('Nontopo_desc')

#%% prepare data for cvss only:
n1cvss=training_nonTopoDatasetArray[:, 23:40]
n2cvss=training_nonTopoDatasetArray[:, 60:]
X_train_cvss = np.concatenate((n1cvss,n2cvss), axis=1)

n1cvss=test_nonTopoDatasetArray[:, 23:40]
n2cvss=test_nonTopoDatasetArray[:, 60:]
X_test_cvss = np.concatenate((n1cvss,n2cvss), axis=1)

L_non.append([X_train_cvss,y_train,X_test_cvss,y_test])
hyper_non.append('Nontopo_cvss')    

#%% following part is about topo features:
#%% prepare data for Top centrality only
dir='./GD_VCBD_Ready_to_go/GD_VCBD_go_CA_Xtopo_centra'
data_cen8 = np.load(dir+".npz",allow_pickle=True)

X_train_cen8=data_cen8["X_train"]
X_test_cen8=data_cen8["X_test"]

L_topo.append([X_train_cen8,y_train,X_test_cen8,y_test])
hyper_topo.append('Topo_centra') 

#%% prepare Topdata community only
dir='./GD_VCBD_Ready_to_go/GD_VCBD_go_CA_Xtopo_commun'
data_com10 = np.load(dir+".npz",allow_pickle=True)

X_train_com10=data_com10["X_train"]
X_test_com10=data_com10["X_test"]

L_topo.append([X_train_com10,y_train,X_test_com10,y_test])
hyper_topo.append('Topo_commun') 

#%% prepare data for topo fastRPEmbedding20 features:
dir='./GD_VCBD_Ready_to_go/GD_VCBD_go_CA_Xtopo_fastRP'
data = np.load(dir+".npz",allow_pickle=True)
X_train_emb20=data["X_train"]
X_test_emb20=data["X_test"]

L_topo.append([X_train_emb20,y_train,X_test_emb20,y_test])
hyper_topo.append('Topo_fastRP') 

# prepare data for 'topo node2vec20 features':
dir='./GD_VCBD_Ready_to_go/GD_VCBD_go_CA_Xtopo_node2vec'
data = np.load(dir+".npz",allow_pickle=True)
X_train_node2vec20=data["X_train"]
X_test_node2vec20=data["X_test"]

L_topo.append([X_train_node2vec20,y_train,X_test_node2vec20,y_test])
hyper_topo.append('Topo_node2vec') 


#%% combine different group of feature:
for N in range(len(L_non)): # append all non-topo
    L.append(L_non[N])
    hyper_list.append(hyper_non[N])

for T in range(len(L_topo)): # append all topo only
    L.append(L_topo[T])
    hyper_list.append(hyper_topo[T])

for N in range(len(L_non)): # combine all
    for T in range(len(L_topo)):
        xtrain= np.concatenate((L_non[N][0],L_topo[T][0]),axis=1)
        xtest= np.concatenate((L_non[N][2],L_topo[T][2]),axis=1)
        L.append([xtrain,L_non[N][1],xtest,L_non[N][3]])
        hyper_list.append(hyper_non[N]+'+'+hyper_topo[T])
        

#%% set parameters for results saving
result_dir="./VCBD_on_CO_AFFECT_subgraph_results.csv"
with open(result_dir, 'a', newline='') as f:
    writer = csv.writer(f)
    my_list = ['Classifier', 'Fearures', 'test_cm','test_acc','test_pre','test_rec','test_f1',
            'test_class1_f1','train_cm','train_acc','train_pre','train_rec','train_f1','train_class1_f1']
    writer.writerow(my_list)
plot_dir=result_dir.replace('results.csv','plot.npy')

#%% VCBD on co_affect subgraph
repeat_times=1
for r in range(repeat_times):
    c_KNN = KNeighborsClassifier(5)
    c_MLP = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(64), activation='logistic', random_state=0,
                         learning_rate='adaptive', max_iter=2000, shuffle=False, early_stopping=True)
    c_LR = LogisticRegression(random_state=0, solver='liblinear', max_iter=2000)
    c_DT = DecisionTreeClassifier(max_depth=10)
    c_RF = RandomForestClassifier(max_depth=10, n_estimators=9, max_features=1)

    c_ABC = AdaBoostClassifier(base_estimator=c_DT,n_estimators=9)
    c_GNB = GaussianNB()

    classifier_list = [c_KNN, c_MLP,c_LR,c_DT,c_RF,c_ABC,c_GNB]
    classifier_name_list = ["KNN", "MLP","LR","DT","RF","ABC",'GNB']
    for h in trange(len(hyper_list)):
        X_train, y_train, X_test, y_test= L[h]
        hyperp_setting= hyper_list[h]
        for c in trange(len(classifier_list)):
            classifier = classifier_list[c]
            classifier_name = classifier_name_list[c]
            print(classifier_name)
    
            # classifier train
            classifier.fit(X_train, y_train)
            # evaluate classifier on the test set
            predictions = classifier.predict(X_test)
            test_confusion_matrix = confusion_matrix(y_test, predictions)
            report = classification_report(y_test, predictions, labels=[0, 1], target_names=['class 0', 'class 1'],
                                           output_dict=True, zero_division=0)  # output_dict=True
            test_acc = report['accuracy']
            test_pre = report['macro avg']['precision']
            test_rec = report['macro avg']['recall']
            test_f1 = report['macro avg']['f1-score']
            test_class1_f1 = report['class 1']['f1-score']
            # % evaluate on the train set
            predictions = classifier.predict(X_train)
            train_confusion_matrix = confusion_matrix(y_train, predictions)
            report = classification_report(y_train, predictions, labels=[0, 1], target_names=['class 0', 'class 1'],
                                           output_dict=True, zero_division=0)  # output_dict=True
            train_acc = report['accuracy']
            train_pre = report['macro avg']['precision']
            train_rec = report['macro avg']['recall']
            train_f1 = report['macro avg']['f1-score']
            train_class1_f1 = report['class 1']['f1-score']

            pos_score = classifier.predict_proba(X_test)[:,1]
            
            fpr, tpr, threshold = roc_curve(y_test,pos_score)
            
            dic_draw.update({str(classifier_name)+'-'+str(hyperp_setting):[fpr,tpr]})
            
            with open(result_dir, 'a', newline='') as f:
                writer = csv.writer(f)
                my_list = [classifier_name, hyperp_setting, test_confusion_matrix, test_acc, test_pre, test_rec, test_f1,
                        test_class1_f1,
                        train_confusion_matrix, train_acc, train_pre, train_rec, train_f1, train_class1_f1]
                writer.writerow(my_list)

np.save(plot_dir,dic_draw)
