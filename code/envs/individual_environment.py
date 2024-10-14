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
df = pd.read_excel("../data/Price_EVLink.xlsx", usecols=[2], names=None)
original_price = df.values
EVLink_price = np.zeros((3984, 1))
for i in range(83):
    for j in range(48):
        EVLink_price[i*48+j] = original_price[j]
EVLink_price = EVLink_price.squeeze().astype('float64')
theta = [1, 1.2, 1.5]
EVLink_price = EVLink_price * theta[2]
eprice_mean = np.mean(EVLink_price)
# EVLink_price:  [0.2909 0.2909 0.2909 ... 0.7343 0.7343 0.7343]
# eprice_mean: 0.74153125

"""data for EV"""
# 3 kinds of EVs with different parameters of DR
m = loadmat("../data/testingdata.mat")
out1 = np.concatenate((m['out1'], m['out1']), axis=1)  # splicing data
for i in range(5):
    out1 = np.concatenate((out1, m['out1']), axis=1)
out2 = np.concatenate((m['out2'], m['out2']), axis=1)
for i in range(5):
    out2 = np.concatenate((out2, m['out2']), axis=1)
out3 = np.concatenate((m['out3'], m['out3']), axis=1)
for i in range(5):
    out3 = np.concatenate((out3, m['out3']), axis=1)
out1 = out1.squeeze().astype('int')     # reduce the dimension
out2 = out2.squeeze().astype('int')
out3 = out3.squeeze().astype('int')
# out1:  [1 1 1 ... 2 2 2] out2:  [1 1 1 ... 2 1 2] out3:  [0 0 0 ... 1 1 1]
# 3 different parking time
deadline = [3, 8, 10]  # h
# mean speed of EV
speed = [5, 8, 10]
# average charging time to fullfill
chargetime = [5, 10, 15]
# 3 different parameters of demand response
beta1 = [-1, -5, -12]
beta2 = [4, 12, 32]

N_S = 4

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
    # for ev in out1
    dem1 = beta1[0]*price_action+beta2[0]
    if dem1 < 0:
        dem1 = 0
    reward += arrivalnum[0] * dem1 * price_action * charge_action
    penalty += 10 * max(0, (dem1-deadline[0])*arrivalnum[0])
    for i in range(arrivalnum[0]):
        residual_demand = demand_update(residual_demand, np.array([dem1, deadline[0]]).reshape((1, 2)))
        # residual_demand : n rows, 2 columns, [demand, parking time]
    # for ev in out2
    dem2 = beta1[1]*price_action+beta2[1]
    if dem2 < 0:
        dem2 = 0
    reward += arrivalnum[1] * dem2 * price_action * charge_action
    penalty += 10 * max(0, (dem2 - deadline[1]) * arrivalnum[1])
    for i in range(arrivalnum[1]):
        residual_demand = demand_update(residual_demand, np.array([dem2, deadline[1]]).reshape((1, 2)))
    # for ev in out3
    dem3 = beta1[2]*price_action+beta2[2]
    if dem3 < 0:
        dem3 = 0
    reward += arrivalnum[2] * dem3 * price_action * charge_action
    penalty += 10 * max(0, (dem3 - deadline[2]) * arrivalnum[2])
    for i in range(arrivalnum[2]):
        residual_demand = demand_update(residual_demand, np.array([dem3, deadline[2]]).reshape((1, 2)))
    return reward, residual_demand, penalty

"""LLF Charging"""
def demand_charge_llf(residual_demand):
    least = residual_demand[:, 1]*5 - residual_demand[:, 0]  # /charge_energy
    order = [operator.itemgetter(0)(t) - 1 for t in sorted(enumerate(least, 1), key=operator.itemgetter(1), reverse=True)]
    residual_demand[order[:N_S], 0] = residual_demand[order[:N_S], 0] - 1
    residual_demand[:, 1] = residual_demand[:, 1] - 1   # parking time -1
    return residual_demand

def env(iternum, action, residual_demand):
    # print("residual_demand: ", residual_demand)
    price_action = action[0]
    charge_action = action[1]
    arrivalnum = [out1[iternum], out2[iternum], out3[iternum]]  # arrival num of EV
    # charge_action[0] = charging rate of CS0 ; charge_action[1] = charging rate of CS1
    '''Demand_charge_LLF:  Charging Station Start to Charge'''
    inum = residual_demand.shape[0]
    # if charge_action[0] > residual_demand.shape[0]:
    #    charge_action[0] = residual_demand.shape[0]
    # if charge_action[1] > residual_demand.shape[0]:
    #    charge_action[1] = residual_demand.shape[0]
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
    ecost = charge_action * charging_num * EVLink_price[iternum]
    # print("reward from evs: ", reward_input)
    # print("cost: ", ecost)
    # print("penalty: ", penalty)
    reward_output = reward_input - ecost - penalty
    # print("final profit", reward_output)
    return reward_output, residual_demand, charging_num

def CSstate(iternum, real_state):
    EVsnum = real_state.shape[0]
    cs_state = [0, 0, 0]
    for i in range(EVsnum):
        cs_state[0] = cs_state[0]+real_state[i][0]
        cs_state[1] = cs_state[1]+real_state[i][1]
    cs_state[2] = EVLink_price[iternum]
    # cs_state = tuple(cs_state)
    return cs_state

class EVenv(object):
    def __init__(self):
        super(EVenv, self).__init__()
        self.action_space = np.arange(18)  # action[0,1,2...,17]
        self.state_dim = 3  # dimension of state
        self.action_dim = self.action_space.shape[0]   # dimension of action:81
        self.EVLink_price = original_price

    def seed(self, seed):
        random.seed(seed)

    def reset(self):
        # state = [2, 3, 0.5, 0.5, 0.2909]  # s = [demand, parking time, prob_cs1, prob_cs2, EVLink_price]
        state = (2, 3, 0.2909)
        real_state = np.array([[2, 3]])
        return real_state, state

    def step(self, iternum, action, real_state):
        # price1: 谷：0.8909, 平：1.4026, 峰：1.8088  ￥/KWh
        # price2: 谷：1.0962，平：1.5343， 峰：2.0290
        # charge_rate: 慢充：3, 交直流母线：5, 快充：7  KWh
        # action_input = [price1,price2,charge_rate1,charge_rate2]
        # prices = [0.8909, 1.0962, 1.4026, 1.5343, 1.8088, 2.0290]
        # prices = [1.0691, 1.6831, 2.1701, 1.3154, 1.8412, 2.4348]
        prices = [1.3364, 2.1039, 2.7132, 1.6443, 2.3015, 3.0435]
        rates = [3, 5, 7]
        actions = {
            0: [prices[0], rates[0]],
            1: [prices[0], rates[1]],
            2: [prices[0], rates[2]],
            3: [prices[1], rates[0]],
            4: [prices[1], rates[1]],
            5: [prices[1], rates[2]],
            6: [prices[2], rates[0]],
            7: [prices[2], rates[1]],
            8: [prices[2], rates[2]],
            9: [prices[3], rates[0]],
            10: [prices[3], rates[1]],
            11: [prices[3], rates[2]],
            12: [prices[4], rates[0]],
            13: [prices[4], rates[1]],
            14: [prices[4], rates[2]],
            15: [prices[5], rates[0]],
            16: [prices[5], rates[1]],
            17: [prices[5], rates[2]],
        }
        action_input = actions.get(action)
        reward, real_state_, charging_num = env(iternum, action_input, real_state)
        cs_state = CSstate(iternum, real_state_)
        return reward, real_state_, cs_state, charging_num