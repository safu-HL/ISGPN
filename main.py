import random
import dgl
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
# K=5
# person_dict=eval(str(np.load("data/person_dict.npy",allow_pickle=True)))
# print(person_dict)

import random
import scipy.stats as ss

x = [65.5172413,65.5172413,65.5172413,75.8620689,68.96551724]

y = [55.17241379,62.0689655,58.6206896,41.3793103,44.827586 ]
# y=[62.0689655,58.62068955,62.0689655,58.62068955,62.0689655]


stats1, p1 = ss.ranksums(x, y, alternative='two-sided')
stats2, p2 = ss.ranksums(x, y, alternative='greater')
stats3, p3 = ss.ranksums(x, y, alternative='less')
print(ss.ranksums(x,y,alternative='two-sided'))

print(stats1)
print(stats2)
print(stats3)

print('p1:', p1)
print('p2:', p2)
print('p3:', p3)

















