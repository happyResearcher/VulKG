
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FuncFormatter
def to_percent(temp, position):
    return '%4.0f'%(100*temp) + '%'




#%% 并列柱形图 GNN comparison
name_list = ['MLP','RF','ABC', 'DT','KNN','LR','GNB']

Fidx=list(range(0,len(name_list)*2,2))
Tidx=list(range(2,len(name_list)*2,2))
F=list([0.6935,0.5837,0.6119,0.6741,0.7076,0.5290,0.5165])
T=list([0.8567,0.8468,0.8448,0.8231,0.8174,0.5557,0.5439])


x = np.arange(len(name_list))  # the label locations
width = 0.40  # the width of the bars

fig, ax = plt.subplots(figsize=(6, 3),dpi=100)
rects1 = ax.bar(x - width/2, F, width, label='Best in Nontopo feature group')
for a,b in zip(x,F):
    ax.text(a- width/2-0.05 , b+0.01, "{:.2%}".format(b), ha='center', va= 'bottom',fontsize=9)

rects2 = ax.bar(x + width/2, T, width, label='Best in Topo feature group')
for a,b in zip(x,T):
    ax.text(a+width/2, b+0.01, "{:.2%}".format(b), ha='center', va= 'bottom',fontsize=9)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 score')
ax.set_xlabel('Classifier')
ax.set_ylim([0.4,1.2])
ax.set_yticks([])
ax.set_xticks(x)
ax.set_xticklabels(name_list)
ax.legend()

# 设置共用x轴
plt.twinx()

Imp_penc=[]
for i in range(0,int(len(name_list))):
    baseline = F[i]
    #Imp_penc = Imp_penc+[(T[i]-baseline)/baseline]
    Imp_penc = Imp_penc+[T[i]-baseline]

x = np.arange(len(name_list))  # the label locations
plt.plot(x,Imp_penc,'-o',color='c',label=r'$\Delta$F1 score')
#plt.yticks([0,0.01,0.5])
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.ylim([0,0.35])
plt.legend(loc=2)
plt.ylabel(r'$\Delta$F1 score')
plt.tight_layout()
plt.savefig('.\compare_groups.pdf', bbox_inches='tight', dpi=600)
plt.show()
