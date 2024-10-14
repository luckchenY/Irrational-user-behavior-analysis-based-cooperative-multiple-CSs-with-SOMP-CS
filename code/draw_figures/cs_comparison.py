import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys


re1 = np.load("../outputs/price_comparison/individual_station/theta2/results/train_rewards.npy")
ma1 = np.load("../outputs/price_comparison/individual_station/theta2/results/train_ma_rewards.npy")

re2 = np.load("../outputs/price_comparison/2stations/theta2/results/train_rewards.npy")
ma2 = np.load("../outputs/price_comparison/2stations/theta2/results/train_ma_rewards.npy")

path = "../outputs/price_comparison/theta2"

plt.figure(figsize=(12, 9))
plt.title("")
plt.xticks(fontname="Times New Roman", fontsize=26)
plt.yticks(fontname="Times New Roman", fontsize=26)
plt.xlabel('Episodes', fontdict={"family": "Times New Roman", "size": 32})
plt.ylabel('Profit (CNY)', fontdict={"family": "Times New Roman", "size": 32})

plt.plot(re1, linewidth=3, color="#ffd6b2", label='Individual Station')
plt.plot(re2, linewidth=3, color="#cbd7e4",  label='Cooperative Stations')
plt.plot(ma1, linewidth=3, color="#fb8402", label='Moving Profit of Individual')
plt.plot(ma2, linewidth=3, color="#023047", label='Moving Profit of Cooperative')

plt.legend(prop={"family": "Times New Roman", "size": 24})
plt.savefig(path)
plt.show()

sys.exit()
'''

path = "../outputs/price_comparison/cs comparison under prices"
plt.figure(figsize=(12, 9))
plt.title("")
plt.xticks(fontname="Times New Roman", fontsize=24)
n=3
X=np.arange(n)+1.0
width=0.25
Y1=[1180, 1000, 730]
Y2=[950, 590, 500]
plt.bar(X-width/2, Y1, width=width, label='Cooperative Stations', facecolor='#023047', edgecolor='white')
plt.bar(X+width/2, Y2, width=width, label='Individual Stations', facecolor='#fb8402', edgecolor='white')

plt.ylabel('Profit after Convergence (CNY)', fontdict={"family": "Times New Roman", "size": 32})
plt.yticks(fontname="Times New Roman", fontsize=26)
plt.xticks([1,2,3], ('Low','Middle','High'), fontname="Times New Roman", fontsize=26)
plt.xlabel('Price Level', fontdict={"family": "Times New Roman", "size": 32})

plt.legend(prop={"family": "Times New Roman", "size": 24})
plt.savefig(path)
plt.show()

sys.exit()
'''