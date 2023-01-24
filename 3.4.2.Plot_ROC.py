import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
from textwrap import fill

list_draw=[] # list for storing  4 group of data. 
list_title=[] # list for storing 4 sub-graph title name.

# roc curve data (fpr,tpr)  for co-exploit prediction task from co-affect graph
plot_dir="./VCBD_on_CO_AFFECT_subgraph_plot.npy"
num1=['MLP-Topo_centra','MLP-Nontopo_all+Topo_centra','MLP-Nontopo_all']
num2=['RF-Topo_commun','RF-Nontopo_cvss+Topo_commun','RF-Nontopo_cvss']
list_draw.append([plot_dir,num1,num2])
list_title.append('on co-affect sub-graph')

#plot parameter:
color_list = ['red','black','blue','orange','brown','pink','purple','brown']
leg_length= 15
fsize=22
lw = 1.5
#plt.figure(figsize=(8, 8))
plt.rcParams['figure.figsize'] = (8.0, 8.0)
fdict={'weight':2,'fontsize':fsize}

#plot all 8 figures:
for p in range(len(list_draw)):
    data= np.load(list_draw[p][0],allow_pickle= True).item()
    title= list_title[p]
    for n in range(2):
        key_list=list_draw[p][n+1]
        
        for k in range(len(key_list)):
            print(key_list[k])
            
            [fpr,tpr]= data[key_list[k]]
            roc_auc = auc(fpr,tpr)
            lab=key_list[k]
            if key_list[k].find('Co_Affect') >=0:
                lab=key_list[k].replace('Co_Affect','Topo')

            elif key_list[k].find('Co_CWEID') >= 0:                
                lab=key_list[k].replace('Co_CWEID','Topo')
                
            plt.plot(fpr, tpr, color=color_list[k],lw=lw, label=fill(lab+'(AUC=%0.3f)' % roc_auc,leg_length))

        plt.tick_params(labelsize=fsize-2)
        plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate',fontsize=fsize)
        plt.ylabel('True Positive Rate',fontsize=fsize)
        plt.title('Random', x=0.8, y=0.65,fontdict=fdict)
        plt.title(title,loc='right',fontdict=fdict)
        plt.legend(loc="lower right",fontsize=fsize)
        plt.tight_layout()
        plt.show()

   
