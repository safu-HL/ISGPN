import torch
import numpy as np
import pandas as pd

import xlsxwriter as xw
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score,auc,roc_curve
# roc_auc=[]
# y_test=[0,1,1,1,0,0,0,1]
# y_score=[0.23,0.4,0.1,0.34,0.5,0.7,0.8,0.7]
# fpr, tpr, thread = roc_curve(y_test, y_score)
# print(fpr)
# print(tpr)
# roc_auc.append(auc(fpr, tpr))
# # 绘图
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.savefig('roc.png',)
# plt.show()
person_dict=eval(str(np.load("data/person_dict.npy",allow_pickle=True)))
print(person_dict)