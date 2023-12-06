import torch
import torch.nn as nn
import dgl.function as fn
import numpy as np
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
import pandas as pd

gamma = -0.1   #γ
zeta = 1.1     #ζ
beta = 0.66    # β
eps = 1e-20
sig = nn.Sigmoid()
const1 = beta*np.log(-gamma/zeta + eps)

def l0_train(logAlpha, min, max):
    U = torch.rand(logAlpha.size()).type_as(logAlpha) + eps  #随机生成相似 边数*5 的矩阵的值
    s = sig((torch.log(U / (1 - U)) + logAlpha) / beta)      #sigmoid(
    s_bar = s * (zeta - gamma) + gamma
    mask = F.hardtanh(s_bar, min, max)    #hardtanh 线性分段函数 当 min<x<max 时， y=x
    return mask   #维度不变    #这个就是z  即邻接矩阵 是否取值

def l0_test(logAlpha, min, max):  #不生成干扰
    s = sig(logAlpha/beta)
    s_bar = s * (zeta - gamma) + gamma
    mask = F.hardtanh(s_bar, min, max)
    return mask

def get_loss2(logAlpha):
    return sig(logAlpha - const1)

class pas_net(nn.Module):
    def __init__(self,pathway_Mask,drug_Mask,RNA_Num,DEmRNA_Nodes,DEmRNA_mask,Pathway_Nodes,Pathway_Hidden_Nodes,Out_Nodes,Drug_Nodes,Drug_Hidden_Nodes):
        super(pas_net,self).__init__()
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax(dim=1)
        self.pathway_Mask=pathway_Mask
        self.drug_Mask=drug_Mask
        self.RNA_Num=RNA_Num
        self.DEmRNA_MASK=DEmRNA_mask

        self.grna_bn=nn.BatchNorm1d(self.RNA_Num)
        # self.feat_drop=nn.Dropout(0.7)
        self.path_drop=nn.Dropout(0.5)
        self.path_hide_drop=nn.Dropout(0.6)
        self.drug_drop=nn.Dropout()
        self.drug_hide_drop=nn.Dropout()

        self.rna_m_sc=nn.Linear(self.RNA_Num,DEmRNA_Nodes)

        self.m_path_sc=nn.Linear(DEmRNA_Nodes,Pathway_Nodes)
        self.path_hidden_sc = nn.Linear(Pathway_Nodes, Pathway_Hidden_Nodes)
        self.path_out_sc = nn.Linear(Pathway_Hidden_Nodes, Out_Nodes)

        self.m_drug_sc=nn.Linear(DEmRNA_Nodes,Drug_Nodes)
        self.drug_hidden_sc=nn.Linear(Drug_Nodes,Drug_Hidden_Nodes)
        self.drug_out_sc=nn.Linear(Drug_Hidden_Nodes,Out_Nodes)

        nn.init.xavier_normal_(self.rna_m_sc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.m_path_sc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.path_hidden_sc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.path_out_sc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.m_drug_sc.weight.data,gain=1.414)
        nn.init.xavier_normal_(self.drug_hidden_sc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.drug_out_sc.weight.data, gain=1.414)
        self.select=0

    def forward(self,x):
        # x=self.feat_drop(x)

        # x_df=pd.DataFrame(x.tolist())
        # x_df.to_csv("x.csv",index=False,header=False)
        # m_sc=pd.DataFrame(self.rna_m_sc.weight.data.tolist())
        # print(self.rna_m_sc.weight.data)
        # m_sc.to_csv("m_sc",index=False,header=False)
        # x = self.sigmoid(self.grna_bn(x))
        # x = self.relu(self.rna_m_sc(x))

        # self.rna_m_sc.weight.data=self.rna_m_sc.weight.data.mul(self.DEmRNA_MASK)
        # x=self.relu(self.rna_m_sc(x))
        self.m_path_sc.weight.data = self.m_path_sc.weight.data.mul(self.pathway_Mask)
        path = self.relu(self.m_path_sc(x))
        # path=self.path_drop(path)
        path = self.relu(self.path_hidden_sc(path))
        if self.select==1:
            self.path_hide_zhi=path
        # path=self.path_hide_drop(path)
        path = self.path_out_sc(path)
        if self.select==1:
            self.path_out_zhi=path

        # # self.m_drug_sc.weight.data = self.m_drug_sc.weight.data.mul(self.drug_Mask)
        # drug = self.relu(self.m_drug_sc(x))
        # drug = self.relu(self.drug_hidden_sc(drug))
        # drug = self.drug_out_sc(drug)
        out = self.softmax(path)
        return out

class ceRNAnet(nn.Module):
    def __init__(self,alpha,bias_l0,g,RNA_Num,DEmRNA_index,DEmRNA_mask,
                 pathway_Mask,drug_Mask,mRNA_Num,DEmRNA_Nodes,Pathway_Nodes,Pathway_Hidden_Nodes,Out_Nodes,Drug_Nodes,Drug_Hidden_Nodes):
        super(ceRNAnet,self).__init__()

        self.sigmoid=nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.l0=1
        self.DEmRNA_index=DEmRNA_index
        self.loss=0
        self.g=g

        self.attn_l=nn.Parameter(torch.Tensor(size=(RNA_Num,1)))
        self.attn_r=nn.Parameter(torch.Tensor(size=(RNA_Num,1)))
        self.bias_l0 = nn.Parameter(torch.FloatTensor([bias_l0]))
        nn.init.xavier_normal_(self.attn_l.data,gain=1.414)  # normal 服从正太分布 mean=0 std=gain*sqrt(2/(in+out)) gain 根据激活函数的增益
        nn.init.xavier_normal_(self.attn_r.data, gain=1.414)

        self.pas_net=pas_net(pathway_Mask,drug_Mask,RNA_Num,DEmRNA_Nodes,DEmRNA_mask,Pathway_Nodes,Pathway_Hidden_Nodes,Out_Nodes,Drug_Nodes,Drug_Hidden_Nodes)

    def pre_process(self,rna):
        mean=torch.mean(rna,dim=0)
        std=torch.std(rna,dim=0)
        return ((rna-mean)/std)

    def forward(self,lnc_data,mi_data,m_data):
        self.loss = 0

        RNA_data=torch.cat([lnc_data,mi_data,m_data],1)
        RNA=self.pre_process(RNA_data).T
        RNA=torch.where(torch.isnan(RNA),torch.full_like(RNA,0),RNA)
        # RNA=self.sigmoid(RNA)
        a_RNA=self.pre_process(RNA_data.T)
        # lnc_data=self.pre_process(lnc_data)
        # mi_data = self.pre_process(mi_data)
        # m_data = self.pre_process(m_data)
        # RNA=torch.cat([lnc_data,mi_data,m_data],1).T

        # print(self.attn_l)
        # print(RNA)
        a1=(a_RNA*self.attn_l).sum(dim=-1).unsqueeze(-1)
        a2=(a_RNA*self.attn_r).sum(dim=-1).unsqueeze(-1)
        # print(self.attn_r)
        # print(a1)
        # print(a2)
        self.g.ndata['RNA']=a_RNA
        self.g.ndata['a1']=a1
        self.g.ndata['a2']=a2
        self.g.apply_edges(self.edge_attention,"__ALL__")
        self.edge_softmax()
        # print(self.g.edata['a'])
        if self.l0==1:
            self.g.apply_edges(self.norm)  # 即每条边取平均的方式
        # print(self.g.ndata['z'])
        # print(self.g.edata['a'])
        self.g.edata['a']=torch.where(torch.isnan(self.g.edata['a']),torch.full_like(self.g.edata['a'],0),self.g.edata['a'])
        edges = self.g.edata['a'].squeeze().nonzero().squeeze()  # 得到不为0的边的索引[]
        # print(self.g.edata['a'])
        # print(edges)
        # self.g.edata['a_drop'] = self.attn_drop(self.g.edata['a'])  # 添加新参数到edata
        num = (self.g.edata['a'] > 0).sum()  #
        self.g.update_all(fn.u_mul_e('RNA', 'a', 'RNA'), fn.sum('RNA', 'RNA'))
        RNA_1=self.g.ndata['RNA']
        self.g.update_all(fn.u_mul_e('RNA', 'a', 'RNA'), fn.sum('RNA', 'RNA'))
        RNA_2=self.g.ndata['RNA']

        # RNA = torch.where(torch.isnan(RNA), torch.full_like(RNA, 0), RNA)
        # RNA_1 = torch.where(torch.isnan(RNA_1), torch.full_like(RNA_1, 0), RNA_1)
        # RNA_2=torch.where(torch.isnan(RNA_2),torch.full_like(RNA_2,0),RNA_2)

        X=(a_RNA+RNA_1+RNA_2).T[:,self.DEmRNA_index]
        out=self.pas_net(X)

        return out


    def edge_attention(self, edges):
        # an edge UDF to compute unnormalized attention values from src and dst 根据src和dst计算非标准化关注值的边缘UDF
        if self.l0 == 0:
            m = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])  #边的起点中ndata key为a1的值 加上 边的终点中ndata key为a2的值
        else:
            tmp = edges.src['a1'] + edges.dst['a2']
            logits = tmp + self.bias_l0   #边数加上bias_l0
            # print(logits)
            if self.training:
                m = l0_train(logits, 0, 1)
            else:
                m = l0_test(logits, 0, 1)
            # print(m)

            self.loss = get_loss2(logits).sum()  #为每条边数据的第一个相加
        return {'a': m}

    def normalize(self, logits):
        self._logits_name = "_logits"
        self._normalizer_name = "_norm"
        self.g.edata[self._logits_name] = logits  # 就是删除掉的a
        self.g.update_all(fn.copy_e(self._logits_name, self._logits_name),
                          fn.sum(self._logits_name, self._normalizer_name))  # 更新 norm 为 每个点的对应指向这个点的边值相加
        return self.g.edata.pop(self._logits_name), self.g.ndata.pop(self._normalizer_name)  # 返回删掉的a 和每个点的对应指向这个点的边值相加

    def edge_softmax(self):  # 对每一条要计算的边 计算softmax 下方的总和为  指向这条边的终点的所有的边

        if self.l0 == 0:
            scores = edge_softmax(self.g, self.g.edata.pop('a'))  # pop 取'a' 并删除掉
        else:
            scores, normalizer = self.normalize(self.g.edata.pop('a'))  # 返回删掉的a 和每个点的对应指向这个点的边值相加
            self.g.ndata['z'] = normalizer  # 取第一个数值  复制到z z为每个点的对应指向这个点的边值相加

        # print(normalizer)
        self.g.edata['a'] = scores  # softmax后 取 每条边 的第一个数据 并将第二维设置为1 为一个三维矩阵

    def norm(self, edges):
        # normalize attention
        a = edges.data['a'] / edges.dst['z']  # 每条边的值 为 这条边数值/这条边对应终点的值
        return {'a': a}




