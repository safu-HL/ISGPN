from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import  torch
from network.DataLoader import load_data,load_data_and_label
''' set device '''
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(1))  # gpu
else:
    device = torch.device("cpu")

dtype = torch.FloatTensor
x_mRNA_train,y_train,m_index= load_data_and_label("../data/train_DEmRNA_stand0.csv", dtype,device)
x_mRNA_valid = load_data("../data/validation_DEmRNA_stand0.csv", dtype,device)
x_mRNA_test = load_data("../data/test_DEmRNA_stand0.csv",dtype,device)
x_lncRNA_train = load_data("../data/train_DElncRNA_stand0.csv", dtype,device)
x_lncRNA_valid,y_valid,lnc_index = load_data_and_label("../data/validation_DElncRNA_stand0.csv", dtype,device)
x_lncRNA_test = load_data("../data/test_DElncRNA_stand0.csv",dtype,device)
x_miRNA_train = load_data("../data/train_DEmiRNA_stand0.csv", dtype,device)
x_miRNA_valid = load_data("../data/validation_DEmiRNA_stand0.csv", dtype,device)
x_miRNA_test,y_test,mi_index = load_data_and_label("../data/test_DEmiRNA_stand0.csv",dtype,device)

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
model=RandomForestClassifier(n_estimators=500,random_state=0,max_features=100,min_samples_leaf=5)
RNA_t=torch.cat([x_mRNA_train,x_mRNA_valid],0)
Y_train=torch.cat([y_train,y_valid],0)

model.fit(x_mRNA_train,y_train[:,1])
prediction=model.predict(x_mRNA_test)
print('准确率：', metrics.accuracy_score(prediction, y_test[:,1]))

