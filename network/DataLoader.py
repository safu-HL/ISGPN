import numpy as np
import pandas as pd
import torch
import dgl
from dgl.nn.pytorch import edge_softmax
import xlsxwriter as xw

def load_allData(path):
	print("loading " + path + "......")
	data = pd.read_csv(path, index_col=0)
	gene_name = data.columns.values.tolist()
	name = data.index.values.tolist()
	gene = data.values.tolist()
	dict = {}
	for i in range(len(name)):
		dict[name[i]] = gene[i]
	return (dict,gene_name)

def load_pathway(path,name,dtype,device):
	print("loading "+path+"......")

	pathway=pd.read_csv(path,index_col=0)
	index = pathway.columns.values.tolist()
	index=[name.index(i) for i in index]

	PATHWAY = torch.from_numpy(pathway.values).type(dtype)
	###if gpu is being used
	if torch.cuda.is_available():
		PATHWAY = PATHWAY.to(device)
	###
	return(PATHWAY,index)

def load_pathway2(path,dtype,device):
	print("loading "+path+"......")

	pathway=pd.read_csv(path,index_col=0)
	index = pathway.columns.values.tolist()
	PATHWAY = torch.from_numpy(pathway.values).type(dtype)
	###if gpu is being used
	if torch.cuda.is_available():
		PATHWAY = PATHWAY.to(device)
	###
	return PATHWAY,index

def create_PPI_g(ceRNA_path,PPI_path,name,device):
	print("creating " + ceRNA_path + " and " + PPI_path +"......")
	g=dgl.DGLGraph()
	# g=g.to(device)
	g.add_nodes(len(name))

	#------------------ceRNA
	ceRNA=open(ceRNA_path,"r")
	line=ceRNA.readline()
	while line:
		a,b=line.split("\t")
		u=name.index(a)
		v=name.index(b[0:-1])
		g.add_edges(u,v)
		line=ceRNA.readline()
	ceRNA.close()

	#------------------PPI
	ppi=pd.read_csv(PPI_path,header=None).values
	all_ppi=len(ppi)
	index_ppi=0
	for i in ppi:
		index_ppi+=1
		if index_ppi%1000==0:
			print(str(index_ppi)+"/"+str(all_ppi)+",has been connected")
		try:
			u=name.index(i[0])
			v=name.index(i[1])
		except:
			continue
		g.add_edges(u,v)
		g.add_edges(v,u)
	return g

def new_ceRNA(ceRNA_path,edata):
	ceRNA = open(ceRNA_path, "r")
	line = ceRNA.readline()
	edges = edata.squeeze().nonzero().squeeze()  # 得到不为0的边的索引[]
	nceRNA = xw.Workbook("../data/5/new_ceRNA_5.xlsx")
	nceRNA_rawTable = nceRNA.add_worksheet("sheet1")
	nceRNA_rawTable.activate()
	i=0
	index=0
	while line:
		if i in edges:
			a, b = line.split("\t")
			nceRNA_rawTable.write(index,0,a)
			nceRNA_rawTable.write(index,1,b)
			nceRNA_rawTable.write(index,2,edata[i][0])
			index+=1
		i+=1
		line = ceRNA.readline()
	ceRNA.close()
	nceRNA.close()
	print("new_ceRNA saved")
	return

def keshihua(model):
	m_path_sc=pd.DataFrame(model.m_path_sc.weight.data.tolist())
	m_path_sc.to_csv("../data/5/m_path_sc.csv",index=False,header=False)
	path_hidden_sc = pd.DataFrame(model.path_hidden_sc.weight.data.tolist())
	path_hidden_sc.to_csv("../data/5/path_hidden_sc.csv", index=False, header=False)
	path_out_sc = pd.DataFrame(model.path_out_sc.weight.data.tolist())
	path_out_sc.to_csv("../data/5/path_out_sc.csv", index=False, header=False)
	return

def keshihuazhi(model,y):
	path_hide_zhi=pd.DataFrame(torch.cat([model.path_hide_zhi,y[:,1].unsqueeze(-1)],1).tolist())
	path_hide_zhi.to_csv("../data/5/path_hide_zhi.csv",index=False,header=False)
	path_out_zhi=pd.DataFrame(torch.cat([model.path_out_zhi,y[:,1].unsqueeze(-1)],1).tolist())
	path_out_zhi.to_csv("../data/5/path_out_zhi.csv",index=False,header=False)
	return