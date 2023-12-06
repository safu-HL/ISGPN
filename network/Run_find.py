import copy
import torch
import torch.optim as optim
import numpy as np
from DataLoader import load_allData,load_pathway,create_PPI_g,new_ceRNA,keshihua,keshihuazhi
from Model import ceRNAnet
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl.data.utils as utils
from Imbalance_CostFunc import binary_cross_entropy_for_imbalance,focal_loss
from sklearn.metrics import roc_auc_score, f1_score,auc,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import scipy.stats as ss

def get_auc(predict,label):
    sum=len(label)
    auc=0
    for i in range(sum):
        if label[i][predict[i].index(max(predict[i]))] == float(1):
            auc+=1
    return ((auc*1.0)/sum)*100

def get_f1(y_true, y_pred):
    ###covert one-hot encoding into integer
    y = torch.argmax(y_true, dim = 1)
    ###estimated targets (either 0 or 1)
    pred = torch.argmax(y_pred, dim = 1)
    ###if gpu is being used, transferring back to cpu
    if torch.cuda.is_available():
        y = y.cpu().detach()
        pred = pred.cpu().detach()
    ###
    f1 = f1_score(y.numpy(), pred.numpy())
    return(f1)

''' set device '''
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(0))  # gpu
else:
    device = torch.device("cpu")

''' ceRNAnet Settings'''

alpha=0.2    #无效果参数
loss_l0=0.1
Pathway_Nodes=1104
Pathway_Hidden_Nodes=500    #500
Drug_Nodes=830              #830    2501   2946
Drug_Hidden_Nodes=150            #150    500   600
Out_Nodes=2

''' Initial Settings '''
nEpochs = 350             #350
batch_size=64             #64  32
lr=0.0001                   #0.001   0.0001
l2=0.0004                   #0.002   0.0004

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
DEmRNA_mask=torch.zeros(len(DEmRNA_index),len(rna_name))
DEmRNA_mask[:,DEmRNA_index]=1
DEmRNA_mask.to(device)

# g=create_PPI_g("../data/"+str(K)+"/input_interaction_"+str(K)+".txt","../data/PPI.csv",rna_name,device)
# utils.save_graphs("GNN_"+str(K)+".bin",g)
# print("saved")
g=utils.load_graphs("GNN_"+str(K)+".bin")[0][0]
g=g.to(device)


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

x_mRNA_train=torch.from_numpy(np.array(x_mRNA_train)).type(dtype).to(device)
x_miRNA_train=torch.from_numpy(np.array(x_miRNA_train)).type(dtype).to(device)
x_lncRNA_train=torch.from_numpy(np.array(x_lncRNA_train)).type(dtype).to(device)
x_mRNA_valid=torch.from_numpy(np.array(x_mRNA_valid)).type(dtype).to(device)
x_miRNA_valid=torch.from_numpy(np.array(x_miRNA_valid)).type(dtype).to(device)
x_lncRNA_valid=torch.from_numpy(np.array(x_lncRNA_valid)).type(dtype).to(device)
x_mRNA_test=torch.from_numpy(np.array(x_mRNA_test)).type(dtype).to(device)
x_miRNA_test=torch.from_numpy(np.array(x_miRNA_test)).type(dtype).to(device)
x_lncRNA_test=torch.from_numpy(np.array(x_lncRNA_test)).type(dtype).to(device)
y_train=list(map(int, y_train))
Y_train=torch.from_numpy(np.eye(2)[y_train]).type(dtype).to(device)
y_valid=list(map(int, y_valid))
Y_valid=torch.from_numpy(np.eye(2)[y_valid]).type(dtype).to(device)
y_test=list(map(int, y_test))
Y_test=torch.from_numpy(np.eye(2)[y_test]).type(dtype).to(device)
''''''

''' model '''
model=ceRNAnet(alpha,0,copy.deepcopy(g),len(rna_name),DEmRNA_index,DEmRNA_mask,
               pathway_mask,"",len(m_name),len(DEmRNA_index),Pathway_Nodes,Pathway_Hidden_Nodes,Out_Nodes,Drug_Nodes,Drug_Hidden_Nodes)
if torch.cuda.is_available():
    model.to(device)
opt = optim.Adam(model.parameters(), lr=lr,weight_decay=l2)

''' train '''

train_loss_plt=[]
train_auc_plt=[]
train_f1_plt=[]
valid_loss_plt=[]
valid_auc_plt=[]
valid_f1_plt=[]
min_val_auc=0
best_model=copy.deepcopy(model)
N=x_mRNA_train.size()[0]  # num of sample
for epoch in range(nEpochs):
    model.train()

    Sample=list(range(N))
    random.shuffle(Sample)
    train_loss=[]
    train_auc=[]
    train_f1=[]
    for batch in range(0,N,batch_size):
        if batch+batch_size<=N:
            Select=Sample[batch:batch+batch_size]
        else:
            Select=Sample[batch:]
        label=Y_train[Select]
        model.g=copy.deepcopy(g)
        opt.zero_grad()
        predict=model(x_lncRNA_train[Select].clone().detach(),x_miRNA_train[Select].clone().detach(),x_mRNA_train[Select].clone().detach())
        # loss=binary_cross_entropy_for_imbalance(predict,label)
        # loss=F.binary_cross_entropy(predict,label)
        loss=2000*focal_loss(predict,label,alpha=0.65,gamma=0.5,reduction="mean")+loss_l0*model.loss
        # print(focal_loss(predict,label,alpha=0.65,gamma=0.5,reduction="mean"))
        # print(loss_l0*model.loss)
        # print(">>")
        loss.backward()
        # print(">>>")
        opt.step()
        train_loss.append(loss.item())
        train_auc.append(get_auc(predict.tolist(),label))
        train_f1.append(get_f1(label,predict))
        # break ################
    train_loss_plt.append(np.array(train_loss).mean())
    train_auc_plt.append((np.array(train_auc).mean()))
    train_f1_plt.append(np.array(train_f1).mean())
    print("train-epoch: "+str(epoch)+" and the loss is "+str(train_loss_plt[-1])+" and the auc is "+str(train_auc_plt[-1])+"% and the f1 is "+str(train_f1_plt[-1]))

    model.eval()
    model.g = copy.deepcopy(g)
    predict=model(x_lncRNA_valid,x_miRNA_valid,x_mRNA_valid)
    loss=F.binary_cross_entropy(predict,Y_valid)+loss_l0 * model.loss
    valid_loss_plt.append(loss.item())
    valid_auc_plt.append(get_auc(predict.tolist(),Y_valid))
    valid_f1_plt.append(get_f1(Y_valid,predict))
    print("valid-epoch: " + str(epoch) + " and the loss is " + str(valid_loss_plt[-1]) + " and the auc is " + str(valid_auc_plt[-1]) + "% and the f1 is "+str(valid_f1_plt[-1]))
    # break ######################
    # if valid_auc_plt[-1]>=min_val_auc:
    #     best_model=copy.deepcopy(model)
    #     min_val_auc=valid_auc_plt[-1]

# model.load_state_dict(best_model.state_dict())

###########  comparation
# model_comparation=SVC(kernel='rbf',C=1,gamma='auto')
# model_comparation=RandomForestClassifier(n_estimators=200,random_state=0,max_features=50,min_samples_leaf=5)
model_comparation = LogisticRegression(penalty='l2',dual=False,C=1.0,n_jobs=1,random_state=20)
model_comparation.fit(x_mRNA_train,y_train)
prediction_comparation=model_comparation.predict(x_mRNA_test)
prediction_comparation=torch.from_numpy(np.eye(2)[prediction_comparation]).type(dtype).to(device)
print(prediction_comparation)
##########

model.eval()
model.g=copy.deepcopy(g)
model.pas_net.select=1
predict=model(x_lncRNA_test,x_miRNA_test,x_mRNA_test)
# new_ceRNA("../data/"+str(K)+"/input_interaction_"+str(K)+".txt",model.g.edata['a'])
# keshihua(model.pas_net)
# keshihuazhi(model.pas_net,Y_test)
loss=F.binary_cross_entropy(prediction_comparation,Y_test)+loss_l0*model.loss
aucc=get_auc(prediction_comparation.tolist(),Y_test)
f1=get_f1(Y_test,prediction_comparation)
fpr, tpr, thread = roc_curve(Y_test.detach().numpy()[:,1], prediction_comparation.detach().numpy()[:,1])
roc_auc = auc(fpr, tpr)
print("the test loss is "+str(loss))
print("the test auc is "+str(aucc)+"%")
print("the test f1 is "+str(f1))

##################
# print(predict)
# print(prediction_comparation)
stats1, p1 = ss.ranksums(predict[:,1].detach().numpy(), prediction_comparation[:,1], alternative='two-sided')
print(stats1)
print(p1)

#################
plt.figure(dpi=600)
lw = 2
plt.plot(fpr, tpr, color='c',
         lw=lw, label='SVM')  #'ROC curve (area = %0.2f)' % roc_auc
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


plt.plot(train_auc_plt)
plt.plot(valid_auc_plt)
plt.savefig("outcome_auc.png")
plt.show()

plt.plot(train_loss_plt)
plt.plot(valid_loss_plt)
plt.savefig("outcome_loss.png")
plt.show()
















