import numpy as np
from openpyxl import load_workbook
import scipy.io
import pandas as pd
import math

y_ticks = []
x_ticks = []  # 自定义横纵轴标签
corrlist=pd.read_csv("../data/5/m_path_sc_backup.csv",index_col=0)
print(corrlist)
gene_name = corrlist.columns.values.tolist()
path_name = corrlist.index.values.tolist()
x_ticks=gene_name[:-3]
y_ticks=path_name[:10]
# corrlist[abs(corrlist)<0.1]=0
# print(corrlist[0:100,0:100])
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
font = {'family' : 'Times New Roman',
'weight' : 'bold',
'size' : 20,
}
plt.figure(figsize=(12,6),dpi=300)
# plt.tick_params(labelsize=6)
# plt.yticks(fontsize=12)
pd.set_option('expand_frame_repr',False)
# df=pd.read_csv(r"D:\柳州\速度报警类型.csv",encoding='utf-8')

# plt.rcParams['font.sans-serif'] = ['SimHei'] #可以正常显示中文
ax = sns.heatmap(corrlist.values[:10,:-3],annot=False,fmt='.0f',cmap="bwr",vmax=0.2, vmin=-0.2,xticklabels=x_ticks,yticklabels=y_ticks)   #xticklabels=x_ticks, yticklabels=y_ticks
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=12)
#  annot=True表示每一个格子显示数字;fmt='.0f'表示保留0位小数，同理fmt='.1f'表示保留一位小数
#  camp表示颜色，在另一个博主的文章中讲解的很清晰
#  vmax=350, vmin=20表示右侧颜色条的最大最小值，在最大最小值外的颜色将直接以最大或最小值的颜色显示，
#  通过此设置就可以解决少数值过大从而使得大部分色块区别不明显的问题
#  xticklabels=x_ticks, yticklabels=y_ticks，横纵轴标签
# ax.set_title('分速度报警类型分布')  # 图标题
# ax.set_xlabel('Genes')  # x轴标题
# ax.set_ylabel('WSI features')  # y轴标题
plt.show()
figure = ax.get_figure()
# figure.savefig('D:\柳州\分速度报警类型.jpg')  # 保存图片

