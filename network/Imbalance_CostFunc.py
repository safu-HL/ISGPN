import torch
import torch.nn.functional as F

def bce_for_one_class(predict, target, lts = False):
	'''calculate cross entropy in average for samples who belong to the same class. 计算属于同一类别的样本的平均交叉熵
	lts = False: non-LTS samples are obtained. 非LTS样本
	'''
	lts_idx = torch.argmax(target, dim = 1)  #每一行最大的索引 即Y
	if lts == False:
		idx = 0 # label = 0, non-LTS
	else: idx = 1 # label = 1, LTS
	y = target[lts_idx == idx]
	pred = predict[lts_idx == idx]
	cost = F.binary_cross_entropy(pred, y)  #二元交叉熵

	return(cost)

def binary_cross_entropy_for_imbalance(predict, target):
	'''claculate cross entropy for imbalance data in binary classification.'''
	total_cost = bce_for_one_class(predict, target, lts = True) + bce_for_one_class(predict, target, lts = False)
	# cost = F.binary_cross_entropy(predict, target)  # 二元交叉熵
	return(total_cost)

def focal_loss(predict,target,alpha=-1.0,gamma=2.0,reduction="sum"):  # alpha 如0.3的1  和0.7 的0
	ce_loss = F.binary_cross_entropy(predict, target, reduction="none")
	p_t = predict * target + (1 - predict) * (1 - target)  # 全部转为标签为1的数据
	loss = ce_loss * ((1 - p_t) ** gamma)

	if alpha >= 0:
		alpha_t = alpha * target + (1 - alpha) * (1 - target)
		loss = alpha_t * loss

	if reduction == "mean":
		loss = loss.mean()
	elif reduction == "sum":
		loss = loss.sum()
	return loss



