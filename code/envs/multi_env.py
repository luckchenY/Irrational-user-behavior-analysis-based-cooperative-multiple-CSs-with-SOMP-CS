from traceback import print_exc
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
# EVLink_price:  [0.2909 0.2909 0.2909 ... 0.7343 0.7343 0.7343]
# eprice_mean: 0.74153125

"""data for EV"""
# 3 kinds of EVs with different parameters of DR
m = loadmat("../../data/testingdata.mat")
#m = loadmat("/home/pc/LIUJIE/RL-Pricing
# -Charging-Schedule/data/testingdata.mat")
out1 = np.concatenate((m['out1'], m['out1']), axis=1)  # splicing data
for i in range(5):
    out1 = np.concatenate((out1, m['out1']), axis=1)
out2 = np.concatenate((m['out2'], m['out2']), axis=1)
for i in range(5):
    out2 = np.concatenate((out2, m['out2']), axis=1)
out3 = np.concatenate((m['out3'], m['out3']), axis=1)
for i in range(5):
    out3 = np.concatenate((out3, m['out3']), axis=1)
out1 = out1.squeeze().astype('int')     # YINGLONGGU
out2 = out2.squeeze().astype('int')   # PENGLITAI
out3 = out3.squeeze().astype('int')   # PUTAOJIU
# out1:  [1 1 1 ... 2 2 2] out2:  [1 1 1 ... 2 1 2] out3:  [0 0 0 ... 1 1 1]  # *5
# remain mileage to get to each CS for each arrival EV
remm = np.array([[5, 20, 13, 12], [18, 2, 6, 32], [12, 25, 10, 10]])  #  YINGLONGGU, PENGLITAI, PUTAOJIU
# mean speed of EV
speed = 8
# 3 different parking time
deadline = [6, 12, 3]  # h  normal, resident,emergent
# 3 different parameters of demand response
beta1 = [-5, -12, -1]
beta2 = [12, 32, 4]

"""data for CS"""
portsnum = [12, 6, 8, 3]  # the num of ports in CS *5
# average charging time to fullfill
servicetime = [3, 12, 8, 10]
N_S = 12

"""Probability Choise Function"""
def probChoise(price_action, evtype):
    '''traveling cost'''
    timecost = [0, 0, 0, 0]
    timecost[0] = remm[evtype][0] / speed
    timecost[1] = remm[evtype][1] / speed
    timecost[2] = remm[evtype][2] / speed
    timecost[3] = remm[evtype][3] / speed
    '''waiting cost'''
    waitcost = [0, 0, 0, 0]
    waitcost[0] = servicetime[0]
    waitcost[1] = servicetime[1]
    waitcost[2] = servicetime[2]
    waitcost[3] = servicetime[3]
    '''charging cost'''
    chargecost = [0, 0, 0, 0]
    chargecost[0] = price_action[0] * (beta1[evtype] * price_action[0] + beta2[evtype])
    chargecost[1] = price_action[1] * (beta1[evtype] * price_action[1] + beta2[evtype])
    chargecost[2] = price_action[2] * (beta1[evtype] * price_action[2] + beta2[evtype])
    chargecost[3] = price_action[3] * (beta1[evtype] * price_action[3] + beta2[evtype])
    '''util of each CS'''
    util = [0, 0, 0, 0]
    util[0] = -(chargecost[0] + timecost[0] + waitcost[0])
    util[1] = -(chargecost[1] + timecost[1] + waitcost[1])
    util[2] = -(chargecost[2] + timecost[2] + waitcost[2])
    util[3] = -(chargecost[3] + timecost[3] + waitcost[3])
    '''probability of choose cs for EV i'''
    probcs = [0, 0, 0, 0]
    probcs[0] = util[0] / (util[0] + util[1] + util[2] + util[3])
    probcs[1] = util[1] / (util[0] + util[1] + util[2] + util[3])
    probcs[2] = util[2] / (util[0] + util[1] + util[2] + util[3])
    probcs[3] = util[3] / (util[0] + util[1] + util[2] + util[3])
    return probcs[0], probcs[1], probcs[2], probcs[3]

# test_action=[0.8909, 1.4026, 1.8088, 1.5789]
# probcs1=probChoise(test_action,0)
# print("test probCS: ", probcs1)
# print("probcs1: ", probcs1[0]) 

"""Demand Update Function"""
def demand_update(current, new):
    # shape[0]:num of rows of matrix ; shape[1]:num of columns of matrix
    if current.shape[0] < 0.5:
        output = new
    else:
        output = np.concatenate((current, new), axis=0)
    return output

"""Demand Response Function(each time slot)"""
def demand_response(price_action, charge_action, residual_demand, arrivalnum):
    reward = 0
    penalty = 0   # penalty for EVs whose demand can not be satisfied before their departure
    probcs1 = probChoise(price_action, 0)
    probcs2 = probChoise(price_action, 1)
    probcs3 = probChoise(price_action, 2)
    # for ev in out1
    dem1 = beta1[0]*(price_action[0]*probcs1[0]+price_action[1]*probcs1[1]+price_action[2]*probcs1[2]+price_action[3]*probcs1[3])+beta2[0]
    if dem1 < 0:
        dem1 = 0
    reward += arrivalnum[0] * dem1 * (price_action[0]*probcs1[0]+price_action[1]*probcs1[1]+price_action[2]*probcs1[2]+price_action[3]*probcs1[3]) * (charge_action[0] *probcs1[0]+charge_action[1]*probcs1[1]+charge_action[2]*probcs1[2]+charge_action[3]*probcs1[3])
    penalty += 10 * max(0, (dem1-deadline[0])*arrivalnum[0])
    for i in range(arrivalnum[0]):
        residual_demand = demand_update(residual_demand, np.array([dem1, deadline[0], probcs1[0], probcs1[1], probcs1[2], probcs1[3]]).reshape((1, 6)))
        # residual_demand : n rows, 6 columns, [demand, parking time, probcs1, probcs2, probcs3, probcs4]
    # for ev in out2
    dem2 = beta1[1]*(price_action[0]*probcs2[0]+price_action[1]*probcs2[1]+price_action[2]*probcs2[2]+price_action[3]*probcs2[3])+beta2[1]
    if dem2 < 0:
        dem2 = 0
    reward += arrivalnum[1] * dem2 * (price_action[0]*probcs2[0]+price_action[1]*probcs2[1]+price_action[2]*probcs2[2]+price_action[3]*probcs2[3]) * (charge_action[0] * probcs2[0]+charge_action[1]*probcs2[1]+charge_action[2]*probcs2[2]+charge_action[3]*probcs2[3])
    penalty += 10 * max(0, (dem2 - deadline[1]) * arrivalnum[1])
    for i in range(arrivalnum[1]):
        residual_demand = demand_update(residual_demand, np.array([dem2, deadline[1], probcs2[0], probcs2[1], probcs2[2], probcs2[3]]).reshape((1, 6)))
    # for ev in out3
    dem3 = beta1[2]*(price_action[0]*probcs3[0]+price_action[1]*probcs3[1]+price_action[2]*probcs3[2]+price_action[3]*probcs3[3])+beta2[2]
    if dem3 < 0:
        dem3 = 0
    reward += arrivalnum[2] * dem3 * (price_action[0]*probcs3[0]+price_action[1]*probcs3[1]+price_action[2]*probcs3[2]+price_action[3]*probcs3[3]) * (charge_action[0] * probcs3[0]+charge_action[1]*probcs3[1]+charge_action[2]*probcs3[2]+charge_action[3]*probcs3[3])
    penalty += 10 * max(0, (dem3 - deadline[2]) * arrivalnum[2])
    for i in range(arrivalnum[2]):
        residual_demand = demand_update(residual_demand, np.array([dem3, deadline[2], probcs3[0], probcs3[1], probcs3[2], probcs3[3]]).reshape((1, 6)))
    return reward, residual_demand, penalty

"""LLF Charging"""
def demand_charge_llf(residual_demand):    # mean_probcs
    least = residual_demand[:, 1]*5 - residual_demand[:, 0]  # /charge_energy
    order = [operator.itemgetter(0)(t) - 1 for t in sorted(enumerate(least, 1), key=operator.itemgetter(1), reverse=True)]
    residual_demand[order[:N_S], 0] = residual_demand[order[:N_S], 0] - 1
    residual_demand[:, 1] = residual_demand[:, 1] - 1   # parking time -1
    return residual_demand

def env(iternum, action, residual_demand):
    # print("residual_demand: ", residual_demand)
    price_action = [action[0], action[1], action[2], action[3]]
    charge_action = [action[4], action[5], action[6], action[7]]
    arrivalnum = [out1[iternum], out2[iternum], out3[iternum]]  # arrival num of EV
    print("arrivalnum:", arrivalnum)
    # charge_action[0] = charging rate of CS0 ; charge_action[1] = charging rate of CS1
    '''Demand_charge_LLF:  Charging Station Start to Charge'''
    mean_probcs = [0, 0, 0, 0]
    mean_probcs1 = [0, 0, 0, 0]
    inum = residual_demand.shape[0]   #[demand, parking time, probcs1, probcs2, probcs3, probcs4]
    if inum != 0:
        for i in range(0, inum):
            mean_probcs1[0] = mean_probcs1[0] + residual_demand[i][2]
            mean_probcs1[1] = mean_probcs1[1] + residual_demand[i][3]
            mean_probcs1[2] = mean_probcs1[2] + residual_demand[i][4]
            mean_probcs1[3] = mean_probcs1[3] + residual_demand[i][5]
        mean_probcs[0] = mean_probcs1[0] / inum
        mean_probcs[1] = mean_probcs1[1] / inum
        mean_probcs[2] = mean_probcs1[2] / inum
        mean_probcs[3] = mean_probcs1[3] / inum
    else:
        mean_probcs[0] = 0
        mean_probcs[1] = 0
        mean_probcs[2] = 0.5
        mean_probcs[3] = 0.5
    # if charge_action[0] > residual_demand.shape[0]:
    #     charge_action[0] = residual_demand.shape[0]
    # if charge_action[1] > residual_demand.shape[0]:
    #     charge_action[1] = residual_demand.shape[0]
    print("mean_probcs: ", mean_probcs)
    if residual_demand.shape[0] > 0.5:
        residual_demand = demand_charge_llf(residual_demand)
    # print("demand_llf: ", residual_demand)
    '''Demand response: EV Admission'''
    reward_input, residual_demand, penalty = demand_response(price_action, charge_action, residual_demand, arrivalnum)
    if residual_demand.shape[0] < 0.5:  # if the queue is empty(shape[0]:rows of matrix)
        return reward_input, residual_demand, inum
    '''Departure: update the residual_demand array(delete EV with dem=0/parking=0)'''
    residual_demand_ = []
    for i in range(residual_demand.shape[0]):
        if residual_demand[i, 1] > 0 and residual_demand[i, 0] > 0:  # EV not finish charging
            residual_demand_.append(residual_demand[i, :])
    residual_demand = np.array(residual_demand_)
    # print("arrival_demand: ", residual_demand)
    '''Calculate Reward'''
    charging_num = inum + arrivalnum[0] + arrivalnum[1] + arrivalnum[2]  # total num of arrival EV and residual EV
    ecost = (charge_action[0] * mean_probcs[0] + charge_action[1] * mean_probcs[1]+ charge_action[2] * mean_probcs[2]+ charge_action[3] * mean_probcs[3]) * charging_num * EVLink_price[iternum]
    print("reward from evs: ", reward_input)
    print("cost: ", ecost)
    print("penalty: ", penalty)
    reward_output = reward_input - ecost - penalty
    print("final profit", reward_output)
    return reward_output, residual_demand, charging_num

test_action = [0.8909, 1.0962, 0.8909,1.0962,3,5,7,5,3,3,5,7]
test_state = np.array([[2, 3, 0.25, 0.25, 0.25, 0.25]])
charging_num=0
for test in range(10):
    test_reward=0
    test_reward, test_state, charging_num = env(test, test_action, test_state)
    print("************test round**************", test)
    #print("test_reward:", test_reward)
    print("test_state: ", test_state)
    print("charging_num: ", charging_num)

def CSstate(iternum, real_state):
    EVsnum = real_state.shape[0]
    cs_state = [0, 0, 0, 0, 0, 0, 0]
    for i in range(EVsnum):
        cs_state[0] = cs_state[0]+real_state[i][0]
        cs_state[1] = cs_state[1]+real_state[i][1]
        cs_state[2] = cs_state[2]+real_state[i][2]
        cs_state[3] = cs_state[3]+real_state[i][3]
        cs_state[4] = cs_state[2]+real_state[i][4]
        cs_state[5] = cs_state[3]+real_state[i][5]
    if EVsnum != 0:
        cs_state[2] = cs_state[2]/EVsnum
        cs_state[3] = cs_state[3]/EVsnum
        cs_state[4] = cs_state[4]/EVsnum
        cs_state[5] = cs_state[5]/EVsnum
    cs_state[6] = EVLink_price[iternum]
    # cs_state = np.array(cs_state)
    return cs_state

def ten_to_three(a):
    i=7
    b=[]
    while i>=0:
        if a>=3**i:
            c=int(a/(3**i))
            b.append(c)
            a-=c*(3**i)
        else:
            b.append(0)
        i-=1
    return b

class EVenv(object):
    def __init__(self):
        super(EVenv, self).__init__()
        self.action_space = np.arange(6561)  # action[0,1,2...,6561]
        self.state_dim = 7  # dimension of state
        self.action_dim = 6561  # dimension of action:81*81
        self.EVLink_price = original_price

    def seed(self, seed):
        random.seed(seed)
        # return seed

    def reset(self):
        state = [2, 3, 0.25, 0.25, 0.25, 0.25, 0.2909]  # s = [demand, parking time, prob_cs1, prob_cs2, prob_cs3, prob_cs4, EVLink_price]
        # state = np.array([2, 3, 0.5, 0.5, 0.2909])
        # state = 0.2909
        real_state = np.array([[2, 3, 0.25, 0.25, 0.25, 0.25]])
        return real_state, state

    def step(self, iternum, action, real_state):
        # price1: 谷：0.8909, 平：1.4026, 峰：1.8088  ￥/KWh
        # price2: 谷：1.0962，平：1.5343， 峰：2.0290
        # charge_rate: 慢充：3, 交直流母线：5, 快充：7  KWh
        # action_input = [price1,price2,price3,price4,charge_rate1,charge_rate2,charge_rate3,charge_rate4]
        prices1 = [0.8909, 1.4026, 1.8088]
        prices2 = [1.0962, 1.5343, 2.0290]
        prices3 = [0.8909, 1.4026, 1.8088]
        prices4 = [1.0962, 1.5343, 2.0290]
        rates = [3, 5, 7]
        dic={}
        dic[0]=prices1
        dic[1]=prices2
        dic[2]=prices3
        dic[3]=prices4
        dic[4]=rates
        dic[5]=rates
        dic[6]=rates
        dic[7]=rates
        actions={}
        for i in range(0,6561):
            ite=ten_to_three(i)
            d=[]
            for j in range(0,8):
                d.append(dic[j][ite[j]])
            actions[i]=d
        print("actions:", actions)
        action_input = actions.get(action)
        reward, real_state_, charging_num = env(iternum, action_input, real_state)
        cs_state = CSstate(iternum, real_state_)
        return reward, real_state_, cs_state, charging_num, EVLink_price[iternum], action_input

