import copy
import torch
import torch.optim as optim
import numpy as np
from DataLoader import load_allData,load_pathway,create_PPI_g,load_pathway2
from Model2 import ceRNAnet2
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl.data.utils as utils
import pandas as pd
from Imbalance_CostFunc import binary_cross_entropy_for_imbalance,focal_loss
import dgl

def get_auc(predict,label):
    sum=len(label)
    auc=0
    for i in range(sum):
        if label[i][predict[i].index(max(predict[i]))] == float(1):
            auc+=1
    return ((auc*1.0)/sum)*100

''' set device '''
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(0))  # gpu
else:
    device = torch.device("cpu")

''' ceRNAnet Settings'''

alpha=0.2    #无效果参数
loss_l0=0.01
Pathway_Nodes=574
Pathway_Hidden_Nodes=200
Drug_Nodes=830              #830    2501   2946
Drug_Hidden_Nodes=150            #150    500   600
Out_Nodes=2

''' Initial Settings '''
nEpochs = 600             #300
batch_size=64             #64  32
lr=0.0001                   #0.001   0.0001
l2=0.0004                   #0.002   0.0004

''' load data  '''
dtype = torch.FloatTensor

data = pd.read_csv("F:/python file/bio network/pas-net/data/train.csv")
x = data.drop(["LTS_LABEL"], axis=1).values  # axis=1删除集合中的列：删除标签列
y = data.loc[:, ["LTS_LABEL"]].values  # 标签列 取一列
y = list(map(int, y))
X = torch.from_numpy(x).type(dtype)
Y = torch.from_numpy(np.eye(2)[y]).type(dtype)
###if gpu is being used
if torch.cuda.is_available():
    X = X.cuda()
    Y = Y.cuda()
###

data_va = pd.read_csv("F:/python file/bio network/pas-net/data/validation.csv")
x_va = data_va.drop(["LTS_LABEL"], axis=1).values  # axis=1删除集合中的列：删除标签列
y_va = data_va.loc[:, ["LTS_LABEL"]].values  # 标签列 取一列
y_va = list(map(int, y_va))
X_va = torch.from_numpy(x_va).type(dtype)
Y_va = torch.from_numpy(np.eye(2)[y_va]).type(dtype)
###if gpu is being used
if torch.cuda.is_available():
    X_va = X_va.cuda()
    Y_va = Y_va.cuda()
###


pathway_mask ,mRNA_name= load_pathway2("F:/python file/bio network/pas-net/data/pathway_mask.csv",dtype,device)

g=dgl.DGLGraph()
g.add_nodes(len(mRNA_name))
a=[random.randint(0,4359) for i in range(100)]
b=[random.randint(0,4359) for i in range(100)]
for i in a:
    for j in b:
        if i!=j:
            g.add_edges(i,j)
g=g.to(device)

# ''' 10k-5fold'''

''' model '''
model=ceRNAnet2(alpha,0,copy.deepcopy(g),len(mRNA_name),"DEmRNA_index","DEmRNA_imask",
               pathway_mask,"",len(mRNA_name),len(mRNA_name),Pathway_Nodes,Pathway_Hidden_Nodes,Out_Nodes,Drug_Nodes,Drug_Hidden_Nodes)
if torch.cuda.is_available():
    model.to(device)
opt = optim.Adam(model.parameters(), lr=lr,weight_decay=l2)

''' train '''

train_loss_plt=[]
train_auc_plt=[]
valid_loss_plt=[]
valid_auc_plt=[]
min_val_auc=0
best_model=copy.deepcopy(model)
N=X.size()[0]  # num of sample
for epoch in range(nEpochs):
    model.train()

    Sample=list(range(N))
    random.shuffle(Sample)
    train_loss=[]
    train_auc=[]
    for batch in range(0,N,batch_size):
        if batch+batch_size<=N:
            Select=Sample[batch:batch+batch_size]
        else:
            Select=Sample[batch:]
        label=Y[Select]
        model.g=copy.deepcopy(g)
        opt.zero_grad()
        predict=model(X[Select].clone().detach())
        # loss=binary_cross_entropy_for_imbalance(predict,label)
        # loss=F.binary_cross_entropy(predict,label)
        loss=focal_loss(predict,label,alpha=0.65,gamma=0.5,reduction="mean")+loss_l0*model.loss
        # print(focal_loss(predict,label,alpha=0.65,gamma=0.5,reduction="mean"))
        # print(loss_l0*model.loss)
        # print(">>")
        loss.backward()
        # print(">>>")
        opt.step()
        train_loss.append(loss.item())
        train_auc.append(get_auc(predict.tolist(),label))
        # break ################
    train_loss_plt.append(np.array(train_loss).mean())
    train_auc_plt.append((np.array(train_auc).mean()))
    print("train-epoch: "+str(epoch)+" and the loss is "+str(train_loss_plt[-1])+" and the auc is "+str(train_auc_plt[-1])+"%")

    model.eval()
    model.g = copy.deepcopy(g)
    predict=model(X_va)
    loss=F.binary_cross_entropy(predict,Y_va)+loss_l0 * model.loss
    valid_loss_plt.append(loss.item())
    valid_auc_plt.append(get_auc(predict.tolist(),Y_va))
    print("valid-epoch: " + str(epoch) + " and the loss is " + str(valid_loss_plt[-1]) + " and the auc is " + str(valid_auc_plt[-1]) + "%")
    # break ######################
    # if valid_auc_plt[-1]>=min_val_auc:
    #     best_model=copy.deepcopy(model)
    #     min_val_auc=valid_auc_plt[-1]

# model.load_state_dict(best_model.state_dict())
# model.eval()
# model.g=copy.deepcopy(g)
# predict=model(x_lncRNA_test,x_miRNA_test,x_mRNA_test)
# loss=F.binary_cross_entropy(predict,Y_test)+loss_l0*model.loss
# auc=get_auc(predict.tolist(),Y_test)
# print("the test loss is "+str(loss))
# print("the test auc is "+str(auc)+"%")

plt.plot(train_auc_plt)
plt.plot(valid_auc_plt)
plt.savefig("outcome_auc.png")
plt.show()

plt.plot(train_loss_plt)
plt.plot(valid_loss_plt)
plt.savefig("outcome_loss.png")
plt.show()
















