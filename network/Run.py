import copy
import torch
import torch.optim as optim
import numpy as np
from DataLoader import load_pathway,load_allData,create_PPI_g
from Model import ceRNAnet
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl.data.utils as utils
from Imbalance_CostFunc import binary_cross_entropy_for_imbalance,focal_loss

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

''' ceRNAnet Settings'''

alpha=0.2    #无效果参数
loss_l0=0.1
Pathway_Nodes=1104
Pathway_Hidden_Nodes=500
Drug_Nodes=830             #830    2501   2946
Drug_Hidden_Nodes=150            #150    500   600
Out_Nodes=2

''' Initial Settings '''
nEpochs = 400
batch_size=64
lr=0.0001                    #0.0001
l2=0.0004                    #0.0004

''' load data  '''
dtype = torch.FloatTensor
K=5
mRNA_dict,m_name=load_allData("../data/"+str(K)+"/mRNA_"+str(K)+".csv")
miRNA_dict,mi_name=load_allData("../data/"+str(K)+"/miRNA_"+str(K)+".csv")
lncRNA_dict,lnc_name=load_allData("../data/"+str(K)+"/lncRNA_"+str(K)+".csv")
rna_name=copy.deepcopy(lnc_name)
rna_name.extend(mi_name)
rna_name.extend(m_name)

pathway_mask,DEmRNA_index = load_pathway("../data/"+str(K)+"/pathwayMask_"+str(K)+".csv",rna_name,dtype,device)
DEmRNA_mask=np.zeros((len(DEmRNA_index),len(rna_name)))
DEmRNA_mask[:,DEmRNA_index]=1
DEmRNA_mask=torch.from_numpy(np.array(DEmRNA_mask)).type(dtype).to(device)

# g=create_PPI_g("../data/"+str(K)+"/input_interaction_"+str(K)+".txt","../data/PPI.csv",rna_name,device)
# utils.save_graphs("GNN_"+str(K)+".bin",g)
# print("saved")
g=utils.load_graphs("GNN_"+str(K)+".bin")[0][0]
g=g.to(device)

''' 10k-5fold '''

dead=np.load("../data/dead_person.npy").tolist()
alive=np.load("../data/alive_person.npy").tolist()
person_dict=eval(str(np.load("../data/person_dict.npy",allow_pickle=True)))
dead_piece=10
alive_piece=19

k_acc=[]
k_loss=[]
kf_acc=[]
kf_loss=[]
for k in range(1):
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
    f_acc=[]
    f_loss=[]
    for f in range(5):
        train_sample = []
        test_sample = []
        for flag in range(10):
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
            x_mRNA_train.append(mRNA_dict[name + "-01"])
            x_miRNA_train.append(miRNA_dict[name + "-01"])
            x_lncRNA_train.append(lncRNA_dict[name + "-01"])
            y_train.append(person_dict[name])
        for name in test_sample:
            x_mRNA_test.append(mRNA_dict[name + "-01"])
            x_miRNA_test.append(miRNA_dict[name + "-01"])
            x_lncRNA_test.append(lncRNA_dict[name + "-01"])
            y_test.append(person_dict[name])
        x_mRNA_train=torch.from_numpy(np.array(x_mRNA_train)).type(dtype).to(device)
        x_miRNA_train=torch.from_numpy(np.array(x_miRNA_train)).type(dtype).to(device)
        x_lncRNA_train=torch.from_numpy(np.array(x_lncRNA_train)).type(dtype).to(device)
        x_mRNA_test=torch.from_numpy(np.array(x_mRNA_test)).type(dtype).to(device)
        x_miRNA_test=torch.from_numpy(np.array(x_miRNA_test)).type(dtype).to(device)
        x_lncRNA_test=torch.from_numpy(np.array(x_lncRNA_test)).type(dtype).to(device)
        y_train=list(map(int, y_train))
        Y_train=torch.from_numpy(np.eye(2)[y_train]).type(dtype).to(device)
        y_test=list(map(int, y_test))
        Y_test=torch.from_numpy(np.eye(2)[y_test]).type(dtype).to(device)

        ''' model '''
        model = ceRNAnet(alpha, 0, copy.deepcopy(g), len(rna_name), DEmRNA_index,DEmRNA_mask,
                         pathway_mask, "", len(m_name), len(DEmRNA_index), Pathway_Nodes, Pathway_Hidden_Nodes,
                         Out_Nodes, Drug_Nodes, Drug_Hidden_Nodes)
        if torch.cuda.is_available():
            model.to(device)
        opt = optim.Adam(model.parameters(), lr=lr,weight_decay=l2)

        ''' train '''
        N = x_mRNA_train.size()[0]  # num of sample
        for epoch in range(nEpochs):
            model.train()

            Sample = list(range(N))
            random.shuffle(Sample)
            for batch in range(0, N, batch_size):
                if batch + batch_size <= N:
                    Select = Sample[batch:batch + batch_size]
                else:
                    Select = Sample[batch:]
                label = Y_train[Select]
                model.g = copy.deepcopy(g)
                opt.zero_grad()
                predict=model(x_lncRNA_train[Select].clone().detach(),x_miRNA_train[Select].clone().detach(),x_mRNA_train[Select].clone().detach())
                loss=2000*focal_loss(predict,label,alpha=0.65,gamma=0.5,reduction="mean")+loss_l0*model.loss
                loss.backward()
                opt.step()
                print("k:"+str(k+1)+" f:"+str(f+1)+" epoch:"+str(epoch+1)+"/"+str(nEpochs)+" train loss:"+str(loss.item())+" train auc:"+str(get_auc(predict.tolist(),label)))

        model.eval()
        model.g = copy.deepcopy(g)
        predict = model(x_lncRNA_test, x_miRNA_test, x_mRNA_test)
        loss=F.binary_cross_entropy(predict,Y_test)+loss_l0 * model.loss
        f_acc.append(get_auc(predict.tolist(),Y_test))
        f_loss.append(loss.item())
        print("k:"+str(k+1)+" f:"+str(f+1)+" test loss:"+str(f_loss[-1])+" test auc:"+str(f_acc[-1]))
    kf_acc.append(f_acc)
    kf_loss.append(f_loss)
    k_acc.append(np.array(f_acc).mean())
    k_loss.append(np.array(f_loss).mean())

for i in range(len(kf_acc)):
    print(kf_acc[i])
    print(kf_loss[i])

print(k_acc)
print(k_loss)

print(np.array(k_acc).mean())









