from mimetypes import read_mime_types
import os
from turtle import width
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import seaborn as sns



re_emergent = np.load("../outputs/dr_comparison/emergent/results/train_rewards.npy")
ma_emergent = np.load("../outputs/dr_comparison/emergent/results/train_ma_rewards.npy")

re_normal = np.load("../outputs/dr_comparison/normal/results/train_rewards.npy")
ma_normal = np.load("../outputs/dr_comparison/normal/results/train_ma_rewards.npy")

re_residual = np.load("../outputs/dr_comparison/residual/results/train_rewards.npy")
ma_residual = np.load("../outputs/dr_comparison/residual/results/train_ma_rewards.npy")

re_mix = np.load("../outputs/dr_comparison/mix_mode/results/train_rewards.npy")
ma_mix = np.load("../outputs/dr_comparison/mix_mode/results/train_ma_rewards.npy")


path = "../outputs/dr_comparison/demand_response"
plt.figure(figsize=(12, 9))
plt.title("")
plt.xticks(fontname="Times New Roman", fontsize=28)
plt.yticks(fontname="Times New Roman", fontsize=28)
plt.xlabel('Episodes', fontdict={"family": "Times New Roman", "size": 32})
plt.ylabel('Moving Reward of Different EV (CNY)', fontdict={"family": "Times New Roman", "size": 32})
plt.plot(re_mix, linewidth=3, color="#b6dafb")
plt.plot(ma_mix, linewidth=3, color="#0c365e", label='Mixed-mode')
plt.plot(re_emergent, linewidth=3, color="#ffeca7")
plt.plot(ma_emergent, linewidth=3, color="#ffdb47", label='Emergent')
plt.plot(re_normal, linewidth=3, color="#f7bdbd")
plt.plot(ma_normal, linewidth=3, color="#c45e63", label='Normal')
plt.plot(re_residual, linewidth=3, color="#b6d3c0")
plt.plot(ma_residual, linewidth=3, color="#107e4d", label='Residual')


plt.legend(loc="lower right", prop={"family": "Times New Roman", "size": 24})
plt.savefig(path)
plt.show()

sys.exit()

'''
path = "../outputs/dr_comparison/profit_after_comparison"
plt.figure(figsize=(12, 9))
plt.title("")

EVs = ['Residual','Mixed-mode','Normal','Emergent']
profits = [1850, 1100, 1000, 500]
width = 0.3
index=np.arange(len(EVs))
plt.bar(index, profits, width, color=["#107e4d","#0c365e","#c45e63","#ffdb47"])
plt.xlabel('Electric Vehicles', fontdict={"family": "Times New Roman", "size": 32})
plt.ylabel('Profit after Convergence (CNY)', fontdict={"family": "Times New Roman", "size": 32})
plt.yticks(fontname="Times New Roman", fontsize=24)
plt.xticks(index, EVs, fontname="Times New Roman", fontsize=30 )

plt.savefig(path)
plt.show()

sys.exit()
'''