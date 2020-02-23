"""
@project = part_time
@file = main.py
@author = xx
@create_time = 2019/09/18 10:33
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# global variables
k = 0  # 样本数
check_non_pre = 0  # 非预防修理器材flag
check_pre = 0  # 预防修理器材flag
stable_or_not_pre = 0  # 预防性稳定序列或者波动序列，波动为0， 稳定为1
stable_or_not_non_pre = 0  # 非预防性稳定序列或者波动序列，波动为0， 稳定为1
combination_check = 0  # 波动序列组合确定，总共有4个组合
non_pre_repair_ep = []  # 非预防性修理消耗序列
pre_repair_ep = []  # 预防性修理消耗序列
repair_ep = []  # 承修装备数量
consume_ep = []  # 器材消耗量序列
repair_ep_es = 0  # 承修设备估计
max_devices_num = 0  # 最大设备型号数量

# 库存上下限
storage_min = 0
storage_max = 0

"""
时间序列确定
"""
def make_sure_list(flag):
	if flag == 0:
		for idx in range(k):
			non_pre_repair_ep.insert(idx + 1, total_data["non"][st_index + idx])
	elif flag == 1:
		max_devices_num = int(input("请输入承修设备一共有多少种型号: "))
		for idx in range(k):
			d = total_data["pre"][idx]
			num_ep_for_repair = total_data_d['devices_num'][idx]
			sum_ep = 0
			for j in range(max_devices_num):
				e = total_data_d["device_{}".format(j + 1)][idx]
				sum_ep += e
			d /= sum_ep
			pre_repair_ep.insert(idx + 1, d)
			repair_ep.insert(idx + 1, sum_ep)
	elif flag == 2:
		for idx in range(k):
			consume_ep.insert(idx + 1, total_data_['consume'][st_index + idx])


"""
读取数据函数
"""


def read_data(type_):
	# 1为预防性数据
	if type_:
		if len(pre_repair_ep) >= 14:
			data = pd.Series(pre_repair_ep[1:14])
		else:
			data = pd.Series(pre_repair_ep)
	# 0为非预防性数据
	else:
		if len(non_pre_repair_ep) >= 14:
			data = pd.Series(non_pre_repair_ep[1:14])
		else:
			data = pd.Series(non_pre_repair_ep)
	y = 100 * data.pct_change().dropna()
	# for j in range(len(y)):
	# 	if y[j] > 1000:
	# 		y[j]*0.1
	return y


"""
进行ARCH过程
"""


def arch(l, type_):
	# 读取数据
	returns = read_data(type_)
	# 处理数据
	am = arch_model(returns, p=l + 1)
	aic = am.fit().aic
	res = am.fit().params
	sum_ = sum(res) - res['mu'] - res['omega'] - res['beta[1]']
	return sum_, aic


"""
波动特征分析
"""


def check_wave_arch(flag):
	min_aic = 21474836
	min_t = 21474836
	optimal = 0
	flac = False
	# 创建ARCH（k）均值方程，并进行性能筛选
	for t in range(1, k + 1):
		if flac:
			break
		if t > 13:
			t = 13
			flac = True
		ss, aic = arch(t, flag)
		if min_aic >= aic:
			min_aic = aic
			optimal = ss
			min_t = t

	print("波动特征分析: {}".format(optimal))
	print("最优的参数个数: {}".format(t))
	if optimal <= 1:
		return True
	else:
		return False


"""
消耗性波动序列库存下限确定
"""


# non_pre = 0, pre = 1
def confirm_wave_min(flag):
	if not flag:
		a = np.argmax(non_pre_repair_ep)
		b = np.argmin(non_pre_repair_ep)
	else:
		a = np.argmax(pre_repair_ep)
		b = np.argmin(pre_repair_ep)
	return (a + b) / 2


"""
库存下限确定
"""


def confirm_min():
	non_pre_min = 0
	pre_min = 0
	if combination_check == 1:
		if check_pre:
			pre_min = np.mean(pre_repair_ep)
			pre_min *= repair_ep_es
		if check_non_pre:
			non_pre_min = np.mean(non_pre_repair_ep)
	elif combination_check == 2:
		non_pre_min = confirm_wave_min(0)
		if check_pre:
			pre_min = np.mean(pre_repair_ep)
			pre_min *= repair_ep_es
	elif combination_check == 3:
		non_pre_min = confirm_wave_min(0)
		pre_min = confirm_wave_min(1)
		pre_min *= repair_ep_es
	elif combination_check == 4:
		pre_min = confirm_wave_min(1)
		pre_min *= repair_ep_es
		if check_non_pre:
			non_pre_min = np.mean(pre_repair_ep)
	return int(non_pre_min + pre_min)


# 函数功能：将频域数据转换成时序数据
# bins为频域数据，n设置使用前多少个频域数据，loop设置生成数据的长度
def fft_combine(bins, n, loops=1):
	length = int(len(bins) * loops)
	data = np.zeros(length)
	index = loops * np.arange(0, length, 1.0) / length * (2 * np.pi)
	for k1, p in enumerate(bins[:n]):
		if k1 != 0:
			p *= 2  # 除去直流成分之外, 其余的系数都 * 2
		data += np.real(p) * np.cos(k1 * index)  # 余弦成分的系数为实数部分
		data -= np.imag(p) * np.sin(k1 * index)  # 正弦成分的系数为负的虚数部分
	return index, data


"""
消耗预测分析
"""


def consume_predict(data_form):
	# 生成随机数
	plt.subplot(2, 1, 1)
	plt.plot(data_form)
	plt.xlabel('time'), plt.ylabel('the num of consumption')

	plt.subplot(2, 1, 2)

	ts_log = np.log(data_form)  # 自然对数为底
	ts_diff = ts_log
	fy = np.fft.fft(ts_diff)
	index, conv2 = fft_combine(fy / len(ts_diff), int(len(fy) / 2 - 1), 1.44)
	ntotal = len(ts_diff) + 3

	plt.plot(np.e ** conv2)
	plt.xticks(np.arange(1, ntotal, 1))
	return np.e ** conv2[ntotal - 2]


"""
库存上限限确定
"""


def confirm_max():
	predict = consume_predict(consume_ep[1:])
	print("今年器材消耗量预测为{}".format(predict))
	return int(storage_min + predict)


if __name__ == "__main__":

	"""
	根据数据得到xi是否是非预防修理器材
	"""
	tmp = input("xi是否是非预防修理器材？")
	if tmp == "y":
		check_non_pre = 1
	else:
		pass

	"""
	根据数据得到xi是否是预防修理器材
	"""
	tmp = input("xi是否是预防修理器材？")
	if tmp == "y":
		check_pre = 1
	else:
		pass

	# 样本K0的数量，及年度数
	st = int(input("请输入样本起始年份： "))
	en = int(input("请输入样本终止年份： "))
	k = en - st
	if check_pre:
		repair_ep_es = int(input("请输入今年度承修设备估计数量： "))

	total_data = pd.read_csv("data_set.csv")
	total_data_ = pd.read_csv("consume_set.csv")
	total_data_d = pd.read_csv("devices_set.csv")
	for i in range(len(total_data['year'])):
		if total_data['year'][i] == st:
			st_index = i

	if check_non_pre:
		make_sure_list(0)  # 时间序列确定
		if check_wave_arch(0):  # 消耗稳定性序列的下限确定
			stable_or_not_non_pre = 1

	if check_pre:
		make_sure_list(1)  # 时间序列确定
		if check_wave_arch(1):  # 消耗稳定性序列的下限确定
			stable_or_not_pre = 1

	"""
	确定特征波动组合
	"""
	if (check_non_pre and stable_or_not_pre) \
			and (check_pre and stable_or_not_non_pre):
		combination_check = 1
	elif (check_non_pre and not stable_or_not_pre) \
			and (check_pre and stable_or_not_non_pre):
		combination_check = 2
	elif (check_non_pre and not stable_or_not_pre) \
			and (check_pre and not stable_or_not_non_pre):
		combination_check = 3
	elif (check_non_pre and stable_or_not_pre) \
			and (check_pre and not stable_or_not_non_pre):
		combination_check = 4
	elif check_non_pre:
		if stable_or_not_non_pre:
			combination_check = 1
		else:
			combination_check = 2
	elif check_pre:
		if stable_or_not_pre:
			combination_check = 1
		else:
			combination_check = 4

	"""
	库存上下限确定
	"""
	storage_min = confirm_min()
	make_sure_list(2)
	storage_max = confirm_max()

	print("今年库存下限为{}".format(storage_min))
	print("今年库存上限为{}".format(storage_max))

	# 显示图像
	plt.grid()
	plt.show()
