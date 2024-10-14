import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(suppress=True)

out1 = [1]
out2 = [1]
out3 = [1]

# 3 different parking time
deadline = [3, 8, 10]  # h
# 3 different parameters of demand response
beta1 = [-1, -5, -12]
beta2 = [4, 12, 32]

remm = np.array([[5, 20], [3, 7], [17, 2]])
# mean speed of EV
speed = [5, 8, 10]
# average charging time to fullfill
chargetime = [5, 10, 15]

"""data for CS"""
portsnum = [12, 30]  # the num of ports in CS
apara = 0.3  # a parameters of attractiveness function
bpara = [0.6, 0.9, 0.3]  # b parameters of attractiveness function

# service price
prices1 = [0.8909, 1.0691, 1.3364, 1.4026, 1.6831, 1.8088, 2.1039, 2.1701, 2.7132]
prices2 = [1.0962, 1.3154, 1.5343, 1.6443, 1.8412, 2.0290, 2.3015, 2.4348, 3.0435]

"""Probability Choise Function"""
def probChoise(price_action, evtype):
    '''attractiveness of CS'''
    attrac = [0, 0]
    attrac[0] = apara * portsnum[0] - bpara[evtype] * price_action[0]  # action[0]=price of CS0
    attrac[1] = apara * portsnum[1] - bpara[evtype] * price_action[1]  # action[1]=price of CS1
    '''the time to reach CS'''
    timecost = [0, 0]
    timecost[0] = remm[evtype][0] / speed[evtype] + chargetime[evtype]
    timecost[1] = remm[evtype][1] / speed[evtype] + chargetime[evtype]
    '''util of each CS'''
    util = [0, 0]
    util[0] = attrac[0] / timecost[0]
    util[1] = attrac[1] / timecost[1]
    '''probability of choose cs for EV i'''
    probcs = [0, 0]
    probcs[0] = util[0] / (util[0] + util[1])
    probcs[1] = util[1] / (util[0] + util[1])
    return probcs[0], probcs[1]

"""Demand Response Function(each time slot)"""
def demand_response(price_action):
    probcs1 = probChoise(price_action, 0)
    probcs2 = probChoise(price_action, 1)
    probcs3 = probChoise(price_action, 2)
    # for ev in out1
    dem1 = beta1[0]*(price_action[0]*probcs1[0]+price_action[1]*probcs1[1])+beta2[0]
    # for ev in out2
    dem2 = beta1[1]*(price_action[0]*probcs2[0]+price_action[1]*probcs2[1])+beta2[1]
    # for ev in out3
    dem3 = beta1[2]*(price_action[0]*probcs3[0]+price_action[1]*probcs3[1])+beta2[2]
    return dem1, dem2, dem3

if __name__ == "__main__":
    demand1 = []
    demand2 = []
    demand3 = []
    for i in range(9):
        price_action = [prices1[i], prices2[0]]
        dem1, dem2, dem3 = demand_response(price_action)
        demand1.append(dem1)
        demand2.append(dem2)
        demand3.append(dem3)
    xname = ['0.8909\n1.0962', '1.0691\n1.3154', '1.3364\n1.5343', '1.4026\n1.6443',
             '1.6831\n1.8412', '1.8088\n2.0290', '2.1039\n2.3015', '2.1701\n2.4348', '2.7132\n3.0435']
    index = np.arange(len(xname))
    sns.set(style='darkgrid')
    plt.figure(figsize=(12, 9))
    plt.xticks(index, xname, fontname="Times New Roman", fontsize=18)
    plt.yticks(fontname="Times New Roman", fontsize=24)
    plt.xlabel('Service Prices of two stations(RMB/KWh)', fontdict={"family": "Times New Roman", "size": 28})
    plt.ylabel('Demand Response(KWh)', fontdict={"family": "Times New Roman", "size": 32})
    plt.plot(demand1, linewidth=3, color="#ffdb47", marker="*", markersize=8, label='Emergent')
    plt.plot(demand2, linewidth=3, color="#c45e63", marker="^", markersize=8, label='Normal')
    plt.plot(demand3, linewidth=3, color="#107e4d", marker="o", markersize=8, label='Residual')

    plt.legend(prop={"family": "Times New Roman", "size": 24})
    plt.savefig("../outputs/comparison/DR")
    plt.show()
    sys.exit()

