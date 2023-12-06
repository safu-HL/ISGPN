import copy
import torch
import torch.optim as optim
import numpy as np
from network.DataLoader import load_data,load_data_and_label,load_ceData,create_PPI_g,load_ceLink,load_pathway,load_pathway2,load_allData
from network.Model import ceRNAnet,regu
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl.data.utils as utils
from network.Imbalance_CostFunc import binary_cross_entropy_for_imbalance,focal_loss
from sklearn.svm import SVC
from sklearn import metrics

def get_auc(predict,label):
    sum=len(label)
    auc=0
    for i in range(sum):
        if label[i][predict[i].index(max(predict[i]))] == float(1):
            auc+=1
    return ((auc*1.0)/sum)*100

''' set device '''
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(2))  # gpu
else:
    device = torch.device("cpu")

dtype = torch.FloatTensor
mRNA_dict,m_index=load_allData("../data/DEmRNA_stand0.csv")
miRNA_dict,mi_index=load_allData("../data/DEmiRNA_stand0.csv")
lncRNA_dict,lnc_index=load_allData("../data/DElncRNA_stand0.csv")

''' 10k-5fold '''

dead=np.load("../data/dead_person.npy").tolist()
alive=np.load("../data/alive_person.npy").tolist()
dead_piece=24
alive_piece=44

k_acc=[]
k_loss=[]
kf_acc=[]
kf_loss=[]
for k in range(10):
    random.shuffle(dead)
    random.shuffle(alive)
    deads=[]
    alives=[]
    for index in range(0,117,dead_piece):
        if index+dead_piece<=117:
            deads.append(dead[index:index+dead_piece])
        else:
            deads.append(dead[index:])
    for index in range(0,217,alive_piece):
        if index+alive_piece<=217:
            alives.append(alive[index:index+alive_piece])
        else:
            alives.append(alive[index:])
    f_acc=[]
    f_loss=[]
    for f in range(5):
        train_sample = []
        test_sample = []
        for flag in range(5):
            if f!=flag:
                train_sample.extend(deads[flag])
                train_sample.extend(alives[flag])
            else:
                test_sample.extend(deads[flag])
                test_sample.extend(alives[flag])
        random.shuffle(train_sample)
        random.shuffle(test_sample)

        x_mRNA_train=[]
        x_miRNA_train=[]
        x_lncRNA_train=[]
        y_train=[]
        x_mRNA_test=[]
        x_miRNA_test=[]
        x_lncRNA_test=[]
        y_test=[]
        for name in train_sample:
            x_mRNA_train.append(mRNA_dict[name][0:-1])
            x_miRNA_train.append(miRNA_dict[name][0:-1])
            x_lncRNA_train.append(lncRNA_dict[name][0:-1])
            y_train.append(miRNA_dict[name][-1])
        for name in test_sample:
            x_mRNA_test.append(mRNA_dict[name][0:-1])
            x_miRNA_test.append(miRNA_dict[name][0:-1])
            x_lncRNA_test.append(lncRNA_dict[name][0:-1])
            y_test.append(miRNA_dict[name][-1])
        y_train=np.eye(2)[list(map(int, y_train))]
        y_test=np.eye(2)[list(map(int, y_test))]

        ''' model '''
        model = SVC(kernel='rbf', C=1, gamma='auto')
        model.fit(x_mRNA_train, y_train[:, 1])
        prediction = model.predict(x_mRNA_test)
        acc=metrics.accuracy_score(prediction, y_test[:, 1])
        print('准确率：', metrics.accuracy_score(prediction, y_test[:, 1]))
        f_acc.append(acc)

    kf_acc.append(f_acc)
    k_acc.append(np.array(f_acc).mean())

for i in range(len(kf_acc)):
    print(kf_acc[i])

print(k_acc)

print(np.array(k_acc).mean())



