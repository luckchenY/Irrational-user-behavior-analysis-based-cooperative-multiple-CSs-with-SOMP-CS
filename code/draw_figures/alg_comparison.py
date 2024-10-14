import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

re_dqn = np.load("../outputs/seed/mean_results/10seed/mean_resultsmean_results.npy")
ma_dqn = np.load("../outputs/seed/mean_results/10seed/mean_resultsmean_ma_results.npy")

re_pg = np.load("../outputs/Policy_Gradient/mean_results/mean_resultsmean_results.npy")
ma_pg = np.load("../outputs/Policy_Gradient/mean_results/mean_resultsmean_ma_results.npy")

re_ql = np.load("../outputs/Q-Learning/10seeds/results/train_rewards.npy")
ma_ql = np.load("../outputs/Q-Learning/10seeds/results/train_ma_rewards.npy")


re_dqn_clip = np.zeros(30)
re_pg_clip = np.zeros(30)
re_ql_clip = np.zeros(30)

for i in range(30):
    re_dqn_clip[i] = re_dqn[i*10]
    re_pg_clip[i] = re_pg[i*10]
    re_ql_clip[i] = re_ql[i*10]



ma_dqn_clip = np.zeros(30)
ma_pg_clip = np.zeros(30)
ma_ql_clip = np.zeros(30)

for i in range(30):
    ma_dqn_clip[i] = ma_dqn[i*10]
    ma_pg_clip[i] = ma_pg[i*10]
    ma_ql_clip[i] = ma_ql[i*10]


# path = "../outputs/comparison/reward_clip_comparison"
#path = "../outputs/comparison/moving_average_reward_comparison"
path = "../outputs/comparison/reward_ma_reward_comparison"

plt.figure(figsize=(12, 9))  # 创建一个图形实例，方便同时多画几个图
plt.xticks(fontname="Times New Roman", fontsize=26)
plt.yticks(fontname="Times New Roman", fontsize=26)
plt.xlabel('Episodes', fontdict={"family": "Times New Roman", "size": 32})
# plt.xlabel('Episodes (x20)', fontdict={"family": "Times New Roman", "size": 32})


'''
plt.ylabel('Average Profit (CNY)', fontdict={"family": "Times New Roman", "size": 32})
plt.plot(re_dqn_clip, linewidth=3, color="#0c365e", marker="*", markersize=5, label='Deep Q-Learning')
plt.plot(re_pg_clip, linewidth=3, color="#780001e", marker="^", markersize=5, label='Policy Gradient')
plt.plot(re_ql_clip, linewidth=3, color="#fb8402", marker="o", markersize=5,  label='Q-Leaning')

'''
'''
plt.ylabel('Moving Average Profit (CNY)', fontdict={"family": "Times New Roman", "size": 32})
plt.plot(ma_dqn_clip, linewidth=3, color="#0c365e", marker="*", markersize=12, label='Deep Q-Learning')
plt.plot(ma_pg_clip, linewidth=3, color="#780001", marker="^", markersize=12, label='Policy Gradient')
plt.plot(ma_ql_clip, linewidth=3, color="#fb8402", marker="o", markersize=12,  label='Q-Leaning')
'''


plt.ylabel('Profit (CNY)', fontdict={"family": "Times New Roman", "size": 32})
plt.plot(re_dqn, linewidth=3, color="#cad7e9")
plt.plot(re_pg, linewidth=3, color="#eabcb3")
plt.plot(re_ql, linewidth=3, color="#ffd6b2")
plt.plot(ma_dqn, linewidth=3, color="#0c365e", label='Deep Q-Learning')
plt.plot(ma_pg, linewidth=3, color="#780001", label='Policy Gradient')
plt.plot(ma_ql, linewidth=3, color="#fb8402",  label='Q-Leaning')


plt.legend(prop={"family": "Times New Roman", "size": 24})
plt.savefig(path)
plt.show()

sys.exit()
