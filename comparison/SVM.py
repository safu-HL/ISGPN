from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import  torch
import copy
import numpy as np
import random
from network.DataLoader import load_allData,load_pathway
from sklearn.metrics import roc_auc_score, f1_score,auc,roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def get_f1(y_true, y_pred):
    ###covert one-hot encoding into integer
    # y = torch.argmax(y_true, dim = 1)
    # ###estimated targets (either 0 or 1)
    # pred = torch.argmax(y_pred, dim = 1)
    ###if gpu is being used, transferring back to cpu
    # if torch.cuda.is_available():
    #     y = y.cpu().detach()
    #     pred = pred.cpu().detach()
    # ###
    f1 = f1_score(y_true, y_pred)
    return(f1)


''' set device '''
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(1))  # gpu
else:
    device = torch.device("cpu")

dtype = torch.FloatTensor
K=5
mRNA_dict,m_name=load_allData("../data/"+str(K)+"/mRNA_"+str(K)+".csv")
miRNA_dict,mi_name=load_allData("../data/"+str(K)+"/miRNA_"+str(K)+".csv")
lncRNA_dict,lnc_name=load_allData("../data/"+str(K)+"/lncRNA_"+str(K)+".csv")
rna_name=copy.deepcopy(lnc_name)
rna_name.extend(mi_name)
rna_name.extend(m_name)

''' 10k-5fold'''
dead=np.load("../data/dead_person.npy").tolist()
alive=np.load("../data/alive_person.npy").tolist()
person_dict=eval(str(np.load("../data/person_dict.npy",allow_pickle=True)))
dead_piece=10
alive_piece=19

random.shuffle(dead)
random.shuffle(alive)
deads=[]
alives=[]
for index in range(0,100,dead_piece):
    if index+dead_piece<=100:
        deads.append(dead[index:index+dead_piece])
    else:
        deads.append(dead[index:])
for index in range(0,188,alive_piece):
    if index+alive_piece<=188:
        alives.append(alive[index:index+alive_piece])
    else:
        alives.append(alive[index:])

train_sample = []
valid_sample=[]
test_sample = []
for flag in range(10):
    if flag==0:
        valid_sample.extend(deads[flag])
        valid_sample.extend(alives[flag])
    elif flag==1:
        test_sample.extend(deads[flag])
        test_sample.extend(alives[flag])
    else:
        train_sample.extend(deads[flag])
        train_sample.extend(alives[flag])
random.shuffle(train_sample)
random.shuffle(valid_sample)
random.shuffle(test_sample)

x_mRNA_train=[]
x_miRNA_train=[]
x_lncRNA_train=[]
y_train=[]
x_mRNA_valid=[]
x_miRNA_valid=[]
x_lncRNA_valid=[]
y_valid=[]
x_mRNA_test=[]
x_miRNA_test=[]
x_lncRNA_test=[]
y_test=[]
for name in train_sample:
    x_mRNA_train.append(mRNA_dict[name+"-01"])
    x_miRNA_train.append(miRNA_dict[name+"-01"])
    x_lncRNA_train.append(lncRNA_dict[name+"-01"])
    y_train.append(person_dict[name])
for name in valid_sample:
    x_mRNA_valid.append(mRNA_dict[name+"-01"])
    x_miRNA_valid.append(miRNA_dict[name+"-01"])
    x_lncRNA_valid.append(lncRNA_dict[name+"-01"])
    y_valid.append(person_dict[name])
for name in test_sample:
    x_mRNA_test.append(mRNA_dict[name+"-01"])
    x_miRNA_test.append(miRNA_dict[name+"-01"])
    x_lncRNA_test.append(lncRNA_dict[name+"-01"])
    y_test.append(person_dict[name])

#----------------------多种数据
# RNA_train = torch.cat([x_mRNA_train, x_miRNA_train, x_lncRNA_train], 1)
# RNA_valid=torch.cat([x_mRNA_valid,x_miRNA_valid,x_lncRNA_valid],1)
# RNA_t=torch.cat([RNA_train,RNA_valid],0)
# Y_train=torch.cat([y_train,y_valid],0)
# RNA_test=torch.cat([x_mRNA_test,x_miRNA_test,x_lncRNA_test],1)
#
# model=SVC(kernel='rbf',C=1,gamma='auto')
# model.fit(RNA_t,Y_train[:,1])
# prediction=model.predict(RNA_test)
# print('准确率：', metrics.accuracy_score(prediction, y_test[:,1]))

#--------------------单mRNA
# model=SVC(kernel='rbf',C=1,gamma='auto')
model=RandomForestClassifier(n_estimators=200,random_state=0,max_features=50,min_samples_leaf=5)
# model = LogisticRegression(penalty='l2',dual=False,C=1.0,n_jobs=1,random_state=20)
model.fit(x_mRNA_train,y_train)
prediction=model.predict(x_mRNA_test)
print(prediction.tolist())
print(y_test)
f1=get_f1(y_test,prediction)
print(f1)
fpr, tpr, thread = roc_curve(y_test, prediction.tolist())
roc_auc = auc(fpr, tpr)
print('准确率：', metrics.accuracy_score(prediction, y_test))

plt.figure(dpi=600)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='SGPN')  #'ROC curve (area = %0.2f)' % roc_auc
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()