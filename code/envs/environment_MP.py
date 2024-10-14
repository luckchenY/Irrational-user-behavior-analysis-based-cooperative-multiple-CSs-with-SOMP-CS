"""
Author: Yang Chen
System Describe: Expansion issues of charging stations.SCSP AND SCMP.
Action: charging stations decide their charging price and total charging rate
Goal: maximize the aggregate utility for two charging stations.
"""

import numpy as np
import operator

import torch

np.set_printoptions(suppress=True)
import  pandas  as pd
from scipy.io import loadmat
import time
import random
import copy

"""EVLink price: ￥/KWh"""
df = pd.read_excel("../../data/Price_EVLink.xlsx", usecols=[2], names=None)
# df = pd.read_excel("/home/pc/LIUJIE/RL-Pricing-Charging-Schedule/data/Price_EVLink.xlsx", usecols=[2], names=None)
original_price = df.values
EVLink_price = np.zeros((3984, 1))
for i in range(83):
    for j in range(48):
        EVLink_price[i*48+j] = original_price[j]
EVLink_price = EVLink_price.squeeze().astype('float64')
# theta = [1, 1.2, 1.5]
# EVLink_price = EVLink_price * theta[0]
eprice_mean = np.mean(EVLink_price)

"""data for EV"""
"EV晚上到家的时间，即插入充电桩的时间"
shape, loc, scale =75.80200206125943, -282.2359837044014, 6.195019070936336
# 生成样本
def generate_gamma_samples(shape, scale,loc, size, min_val=0, max_val=287):
    samples = []
    while len(samples) < size:
        sample = np.random.gamma(shape, scale)+loc
        if min_val <= sample <= max_val:
            samples.append(sample)
    return np.array(samples)
# Calculate the bin edges for equal-length intervals
interval_edges = np.linspace(0, 287, num=288 )

EV_come_sample=generate_gamma_samples(shape, loc, scale,100)
EV_come1,_ = np.histogram(EV_come_sample, bins=interval_edges)
EV_come_sample=generate_gamma_samples(shape, loc, scale,50)
EV_come2,_ = np.histogram(EV_come_sample, bins=interval_edges)
EV_come_sample=generate_gamma_samples(shape, loc, scale,50)
EV_come3,_ = np.histogram(EV_come_sample, bins=interval_edges)

# 3 different parameters of demand response
beta1 = [-1, -5, -12]
beta2 = [4, 12, 32]
# average charging time to fullfill
chargetime = [60, 120, 15]
# 3 different parking time
deadline = [36, 96, 120]  # time slots

"""data for CS"""
portsnum = [12, 12]  # the num of ports in CS
# Dictionary where keys are spot IDs and values are IDs of occupied spots
piles = {key: [] for key in range(12)}
residual_requirement = {key:[0,0] for key in range (12)}  #每一个key对应一个数组.表示充电桩的两个口的剩余充电任务
laxity={key:[0,0] for key in range(30)} #所有存在的车用它连接的充电口进行标记
# a function to calculate the relative laxity of an EV.
def calculate_relative_laxity(td, t, soc_target, soc_init, e_batt, p_max):
    """
    Calculate the relative laxity of an EV.
    Parameters:
    - td_i: Departure time of EV_i
    - ti: Current time
    - e_batt: EV battery capacity
    - p_max: Maximum power delivered by the charger
    Returns:
    - The relative laxity of the EV
    """
    # Remaining parking time P_ti
    p_t = td - t

    # EV charging duration Delta_ti
    delta = (soc_target - soc_init) * e_batt / p_max  # Equation (3)

    # Relative laxity L_ri
    l_r = (p_t - delta) / delta  # Equation (1)
    return l_r

def select_somc_spot(occupied_spots, residual_requirement):
    # Step 1.a: Find the minimum number of occupations among all spots
    min_occupations = min(len(occupied_spots[x]) for x in occupied_spots)
    # Select all spots that have this minimum number of occupations
    least_occupied_spots = [x for x in occupied_spots if len(occupied_spots[x]) == min_occupations]

    # Step 1.b: 对这些充电桩进行排序，首选剩余电量最少的桩子
    #######需要根据数据格式设定进行调整################
    sorted_spots = sorted(least_occupied_spots,  key=lambda x: sum(residual_requirement[x]))

    # Step 1.c: Instruct the next PEV to connect to the SOMC spot from Step 1.b with the smallest remaining charging requirement
    selected_spot = sorted_spots[0]
    return selected_spot

def select_somc_spot_d(piles,laxity):
    #和所有的充电口的剩余的充电需求进行排序，
    sorted_spots= sorted(piles,key=lambda x: min(laxity[x]))
    selected_spot = sorted_spots[0]
    return selected_spot

def probChoise(price,type):
    if price[0]>price[1]:
        return 0,1
    else:
        return 1,0

"""Demand Update Function"""
def demand_update(current, new):
    # shape[0]:num of rows of matrix ; shape[1]:num of columns of matrix
    if current.shape[0] < 0.5:
        output = new
    else:
        output = np.concatenate((current, new), axis=0)
    return output
 # residual_demand : n rows, 4 columns, [demand, parking time, probcs1, probcs2]
"""Demand Response Function(each time slot)"""
def demand_response(price_action, charge_action, residual_demand, arrivalnum):
    reward = 0
    penalty = 0   # penalty for EVs whose demand can not be satisfied before their departure
    probcs1 = probChoise(price_action, 0)
    probcs2 = probChoise(price_action, 1)
    probcs3 = probChoise(price_action, 2)
    # for ev in out1
    dem1 = beta1[0]*(price_action[0]*probcs1[0]+price_action[1]*probcs1[1])+beta2[0]
    if dem1 < 0:
        dem1 = 0
    reward += arrivalnum[0] * dem1 * (price_action[0]*probcs1[0]+price_action[1]*probcs1[1]) * (charge_action[0] *
                                                                         probcs1[0]+charge_action[1]*probcs1[1])
    penalty += 10 * max(0, (dem1-deadline[0])*arrivalnum[0])
    for i in range(arrivalnum[0]):
        residual_demand = demand_update(residual_demand, np.array([dem1, deadline[0], probcs1[0], probcs1[1]]).reshape((1, 4)))
        # residual_demand : n rows, 4 columns, [demand, parking time, probcs1, probcs2]

    ############### for ev in out2
    dem2 = beta1[1]*(price_action[0]*probcs2[0]+price_action[1]*probcs2[1])+beta2[1]
    if dem2 < 0:
        dem2 = 0
    reward += arrivalnum[1] * dem2 * (price_action[0]*probcs2[0]+price_action[1]*probcs2[1]) * (charge_action[0] *
                                                                         probcs2[0]+charge_action[1]*probcs2[1])
    penalty += 10 * max(0, (dem2 - deadline[1]) * arrivalnum[1])
    for i in range(arrivalnum[1]):
        residual_demand = demand_update(residual_demand, np.array([dem2, deadline[1], probcs2[0], probcs2[1]]).reshape((1, 4)))

    ############### for ev in out3
    dem3 = beta1[2]*(price_action[0]*probcs3[0]+price_action[1]*probcs3[1])+beta2[2]
    if dem3 < 0:
        dem3 = 0
    reward += arrivalnum[2] * dem3 * (price_action[0]*probcs3[0]+price_action[1]*probcs3[1]) * (charge_action[0] *
                                                                         probcs3[0]+charge_action[1]*probcs3[1])
    penalty += 10 * max(0, (dem3 - deadline[2]) * arrivalnum[2])
    for i in range(arrivalnum[2]):
        residual_demand = demand_update(residual_demand, np.array([dem3, deadline[2], probcs3[0], probcs3[1]]).reshape((1, 4)))
    return reward, residual_demand, penalty

"""LLF Charging,通过宽松度确定要给哪些车进行充电，然后计算一个充电站统一的充电速率。"""
def demand_charge_llf(residual_demand):    # mean_probcs
    least = residual_demand[:, 1]*5 - residual_demand[:, 0]  # /charge_energy
    order = [operator.itemgetter(0)(t) - 1 for t in sorted(enumerate(least, 1), key=operator.itemgetter(1), reverse=True)]
    residual_demand[order[:N_S], 0] = residual_demand[order[:N_S], 0] - 1
    residual_demand[:, 1] = residual_demand[:, 1] - 1   # parking time -1
    return residual_demand

def env(iternum,action,residual_demand):
    price_action = [action[0], action[1]]
    charge_action = [action[2], action[3]]
    arrivalnum = [EV_come1[iternum], EV_come2[iternum], EV_come3[iternum]]  # arrival num of EV
    '''Demand_charge_LLF:  Charging Station Start to Charge'''
    mean_probcs = [0, 0]
    mean_probcs1 = [0, 0]
    inum = residual_demand.shape[0]
    if inum != 0:
        for i in range(0, inum):
            mean_probcs1[0] = mean_probcs1[0] + residual_demand[i][2]
            mean_probcs1[1] = mean_probcs1[1] + residual_demand[i][3]
        mean_probcs[0] = mean_probcs1[0] / inum
        mean_probcs[1] = mean_probcs1[1] / inum
    else:
        mean_probcs[0] = 0
        mean_probcs[1] = 1


    if residual_demand.shape[0] > 0.5:
        residual_demand = demand_charge_llf(residual_demand)
    '''Demand response: EV Admission'''
    reward_input, residual_demand, penalty = demand_response(price_action, charge_action, residual_demand, arrivalnum)
    if residual_demand.shape[0] < 0.5:  # if the queue is empty(shape[0]:rows of matrix)
        return reward_input, residual_demand, inum
    '''Departure: update the residual_demand array(delete EV with dem=0/parking=0)'''
    residual_demand_ = []
    # residual_demand = [residual_demand, parking_time, probCS1, probCS2]
    for i in range(residual_demand.shape[0]):
        if residual_demand[i, 1] > 0 and residual_demand[i, 0] > 0:  # EV not finish charging
            residual_demand_.append(residual_demand[i, :])
    residual_demand = np.array(residual_demand_)
    '''Calculate Reward'''
    charging_num = inum + arrivalnum[0] + arrivalnum[1] + arrivalnum[2]  # total num of arrival EV and residual EV
    ecost = (charge_action[0] * mean_probcs[0] + charge_action[1] * mean_probcs[1]) * charging_num * EVLink_price[
        iternum]
    # print("reward from evs: ", reward_input)
    # print("cost: ", ecost)
    # print("penalty: ", penalty)
    reward_output = reward_input - ecost - penalty
    # print("final profit", reward_output)
    return reward_output, residual_demand, charging_num

class EVenv(object):
    def __int__(self):
        super(EVenv, self).__init__()
        self.action_space = np.arange(81)  # action[0,1,2...,80]
        self.state_dim = 5  # dimension of state
        self.action_dim = 81  # dimension of action:81
        self.EVLink_price = original_price

    def seed(self, seed):
        random.seed(seed)
        # return seed

    def reset(self):
        state = [2, 3, 0.5, 0.5, 0.2909]  # s = [demand, parking time, prob_cs1, prob_cs2, EVLink_price]
        real_state = np.array([[2, 3, 0.5, 0.5]])
        return real_state, state

    def step(self, iternum, action, real_state):
        # price1: 谷：0.8909, 平：1.4026, 峰：1.8088  ￥/KWh
        # price2: 谷：1.0962，平：1.5343， 峰：2.0290
        # charge_rate: 慢充：3, 交直流母线：5, 快充：7  KWh
        # action_input = [price1,price2,charge_rate1,charge_rate2]
        prices1 = [0.8909, 1.4026, 1.8088]
        prices2 = [1.0962, 1.5343, 2.0290]
        rates = [3, 5, 7]
        # prices1 = [1.0691, 1.6831, 2.1701]
        # prices2 = [1.3154, 1.8412, 2.4348]
        # prices1 = [1.3364, 2.1039, 2.7132]
        # prices2 = [1.6443, 2.3015, 3.0435]
        actions = {
            0: [prices1[0], prices2[0], rates[0], rates[0]],
            1: [prices1[0], prices2[0], rates[0], rates[1]],
            2: [prices1[0], prices2[0], rates[0], rates[2]],
            3: [prices1[0], prices2[0], rates[1], rates[0]],
            4: [prices1[0], prices2[0], rates[1], rates[1]],
            5: [prices1[0], prices2[0], rates[1], rates[2]],
            6: [prices1[0], prices2[0], rates[2], rates[0]],
            7: [prices1[0], prices2[0], rates[2], rates[1]],
            8: [prices1[0], prices2[0], rates[2], rates[2]],
            9: [prices1[0], prices2[1], rates[0], rates[0]],
            10: [prices1[0], prices2[1], rates[0], rates[1]],
            11: [prices1[0], prices2[1], rates[0], rates[2]],
            12: [prices1[0], prices2[1], rates[1], rates[0]],
            13: [prices1[0], prices2[1], rates[1], rates[1]],
            14: [prices1[0], prices2[1], rates[1], rates[2]],
            15: [prices1[0], prices2[1], rates[2], rates[0]],
            16: [prices1[0], prices2[1], rates[2], rates[1]],
            17: [prices1[0], prices2[1], rates[2], rates[2]],
            18: [prices1[0], prices2[2], rates[0], rates[0]],
            19: [prices1[0], prices2[2], rates[0], rates[1]],
            20: [prices1[0], prices2[2], rates[0], rates[2]],
            21: [prices1[0], prices2[2], rates[1], rates[0]],
            22: [prices1[0], prices2[2], rates[1], rates[1]],
            23: [prices1[0], prices2[2], rates[1], rates[2]],
            24: [prices1[0], prices2[2], rates[2], rates[0]],
            25: [prices1[0], prices2[2], rates[2], rates[1]],
            26: [prices1[0], prices2[2], rates[2], rates[2]],
            27: [prices1[1], prices2[0], rates[0], rates[0]],
            28: [prices1[1], prices2[0], rates[0], rates[1]],
            29: [prices1[1], prices2[0], rates[0], rates[2]],
            30: [prices1[1], prices2[0], rates[1], rates[0]],
            31: [prices1[1], prices2[0], rates[1], rates[1]],
            32: [prices1[1], prices2[0], rates[1], rates[2]],
            33: [prices1[1], prices2[0], rates[2], rates[0]],
            34: [prices1[1], prices2[0], rates[2], rates[1]],
            35: [prices1[1], prices2[0], rates[2], rates[2]],
            36: [prices1[1], prices2[1], rates[0], rates[0]],
            37: [prices1[1], prices2[1], rates[0], rates[1]],
            38: [prices1[1], prices2[1], rates[0], rates[2]],
            39: [prices1[1], prices2[1], rates[1], rates[0]],
            40: [prices1[1], prices2[1], rates[1], rates[1]],
            41: [prices1[1], prices2[1], rates[1], rates[2]],
            42: [prices1[1], prices2[1], rates[2], rates[0]],
            43: [prices1[1], prices2[1], rates[2], rates[1]],
            44: [prices1[1], prices2[1], rates[2], rates[2]],
            45: [prices1[1], prices2[2], rates[0], rates[0]],
            46: [prices1[1], prices2[2], rates[0], rates[1]],
            47: [prices1[1], prices2[2], rates[0], rates[2]],
            48: [prices1[1], prices2[2], rates[1], rates[0]],
            49: [prices1[1], prices2[2], rates[1], rates[1]],
            50: [prices1[1], prices2[2], rates[1], rates[2]],
            51: [prices1[1], prices2[2], rates[2], rates[0]],
            52: [prices1[1], prices2[2], rates[2], rates[1]],
            53: [prices1[1], prices2[2], rates[2], rates[2]],
            54: [prices1[2], prices2[0], rates[0], rates[0]],
            55: [prices1[2], prices2[0], rates[0], rates[1]],
            56: [prices1[2], prices2[0], rates[0], rates[2]],
            57: [prices1[2], prices2[0], rates[1], rates[0]],
            58: [prices1[2], prices2[0], rates[1], rates[1]],
            59: [prices1[2], prices2[0], rates[1], rates[2]],
            60: [prices1[2], prices2[0], rates[2], rates[0]],
            61: [prices1[2], prices2[0], rates[2], rates[1]],
            62: [prices1[2], prices2[0], rates[2], rates[2]],
            63: [prices1[2], prices2[1], rates[0], rates[0]],
            64: [prices1[2], prices2[1], rates[0], rates[1]],
            65: [prices1[2], prices2[1], rates[0], rates[2]],
            66: [prices1[2], prices2[1], rates[1], rates[0]],
            67: [prices1[2], prices2[1], rates[1], rates[1]],
            68: [prices1[2], prices2[1], rates[1], rates[2]],
            69: [prices1[2], prices2[1], rates[2], rates[0]],
            70: [prices1[2], prices2[1], rates[2], rates[1]],
            71: [prices1[2], prices2[1], rates[2], rates[2]],
            72: [prices1[2], prices2[2], rates[0], rates[0]],
            73: [prices1[2], prices2[2], rates[0], rates[1]],
            74: [prices1[2], prices2[2], rates[0], rates[2]],
            75: [prices1[2], prices2[2], rates[1], rates[0]],
            76: [prices1[2], prices2[2], rates[1], rates[1]],
            77: [prices1[2], prices2[2], rates[1], rates[2]],
            78: [prices1[2], prices2[2], rates[2], rates[0]],
            79: [prices1[2], prices2[2], rates[2], rates[1]],
            80: [prices1[2], prices2[2], rates[2], rates[2]]
        }
        action_input = actions.get(action)
        reward, real_state_, charging_num = env(iternum, action_input, real_state)
        #在这个函数里的real-state其实是记录之前的residual demand,新来的车辆的需求会根据电价得到响应
        cs_state = CSstate(iternum, real_state_)
        # cs_state = EVLink_price[iternum]
        return reward, real_state_, cs_state, charging_num, EVLink_price[iternum], action_input