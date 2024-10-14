import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

seed1 = np.load("../outputs/seed/1/results/train_rewards.npy")
seed2 = np.load("../outputs/seed/2/results/train_rewards.npy")
seed3 = np.load("../outputs/seed/3/results/train_rewards.npy")
seed4 = np.load("../outputs/seed/4/results/train_rewards.npy")
seed5 = np.load("../outputs/seed/5/results/train_rewards.npy")
seed6 = np.load("../outputs/seed/6/results/train_rewards.npy")
seed7 = np.load("../outputs/seed/7/results/train_rewards.npy")
seed8 = np.load("../outputs/seed/8/results/train_rewards.npy")
seed9 = np.load("../outputs/seed/9/results/train_rewards.npy")
seed10 = np.load("../outputs/seed/10/results/train_rewards.npy")
'''
seed11 = np.load("../code/outputs/seed/11/results/train_rewards.npy")
seed12 = np.load("../code/outputs/seed/12/results/train_rewards.npy")
seed13 = np.load("../code/outputs/seed/13/results/train_rewards.npy")
seed14 = np.load("../code/outputs/seed/14/results/train_rewards.npy")
seed15 = np.load("../code/outputs/seed/15/results/train_rewards.npy")
seed16 = np.load("../code/outputs/seed/16/results/train_rewards.npy")
seed17 = np.load("../code/outputs/seed/17/results/train_rewards.npy")
seed18 = np.load("../code/outputs/seed/18/results/train_rewards.npy")
seed19 = np.load("../code/outputs/seed/19/results/train_rewards.npy")
seed20 = np.load("../code/outputs/seed/20/results/train_rewards.npy")
'''

ma1 = np.load("../outputs/seed/1/results/train_ma_rewards.npy")
ma2 = np.load("../outputs/seed/2/results/train_ma_rewards.npy")
ma3 = np.load("../outputs/seed/3/results/train_ma_rewards.npy")
ma4 = np.load("../outputs/seed/4/results/train_ma_rewards.npy")
ma5 = np.load("../outputs/seed/5/results/train_ma_rewards.npy")
ma6 = np.load("../outputs/seed/6/results/train_ma_rewards.npy")
ma7 = np.load("../outputs/seed/7/results/train_ma_rewards.npy")
ma8 = np.load("../outputs/seed/8/results/train_ma_rewards.npy")
ma9 = np.load("../outputs/seed/9/results/train_ma_rewards.npy")
ma10 = np.load("../outputs/seed/10/results/train_ma_rewards.npy")
'''
ma11 = np.load("../code/outputs/seed/11/results/train_ma_rewards.npy")
ma12 = np.load("../code/outputs/seed/12/results/train_ma_rewards.npy")
ma13 = np.load("../code/outputs/seed/13/results/train_ma_rewards.npy")
ma14 = np.load("../code/outputs/seed/14/results/train_ma_rewards.npy")
ma15 = np.load("../code/outputs/seed/15/results/train_ma_rewards.npy")
ma16 = np.load("../code/outputs/seed/16/results/train_ma_rewards.npy")
ma17 = np.load("../code/outputs/seed/17/results/train_ma_rewards.npy")
ma18 = np.load("../code/outputs/seed/18/results/train_ma_rewards.npy")
ma19 = np.load("../code/outputs/seed/19/results/train_ma_rewards.npy")
ma20 = np.load("../code/outputs/seed/20/results/train_ma_rewards.npy")
'''

mean_results = np.zeros(1000)
mean_ma_results = np.zeros(1000)


for j in range(1000):
    mean_results[j] = (seed1[j]+seed2[j]+seed3[j]+seed4[j]+seed5[j]+seed6[j]+seed7[j]+seed8[j]+seed9[j]+seed10[j] )/10
            # +seed11[j]+seed12[j]+seed13[j]+seed14[j]+seed15[j]+seed16[j]+seed17[j]+seed18[j]+seed19[j]+seed20[j])/20
    mean_ma_results[j] = (ma1[j]+ma2[j]+ma3[j]+ma4[j]+ma5[j]+ma6[j]+ma7[j]+ma8[j]+ma9[j]+ma10[j] )/10
                    # +ma11[j]+ma12[j]+ma13[j]+ma14[j]+ma15[j]+ma16[j]+ma17[j]+ma18[j]+ma19[j]+ma20[j]) / 20

path = "../outputs/comparison/mean_results_10_seeds"
# np.save(path+'mean_results.npy', mean_results)
# np.save(path+'mean_ma_results.npy', mean_ma_results)

plt.figure(figsize=(12, 9))
plt.xticks(fontname="Times New Roman", fontsize=26)
plt.yticks(fontname="Times New Roman", fontsize=26)
plt.xlabel('Episodes', fontdict={"family": "Times New Roman", "size": 32})
plt.ylabel('Average Reward of 10 Different Seeds', fontdict={"family": "Times New Roman", "size": 32})
plt.plot(seed1, linewidth=3, color="#c1dee8")
plt.plot(seed2, linewidth=3, color="#c1dee8")
plt.plot(seed3, linewidth=3, color="#c1dee8")
plt.plot(seed4, linewidth=3, color="#c1dee8")
plt.plot(seed5, linewidth=3, color="#c1dee8")
plt.plot(seed6, linewidth=3, color="#c1dee8")
plt.plot(seed7, linewidth=3, color="#c1dee8")
plt.plot(seed8, linewidth=3, color="#c1dee8")
plt.plot(seed9, linewidth=3, color="#c1dee8")
plt.plot(seed10, linewidth=3, color="#c1dee8", label='Reward with Different Seed')
plt.plot(ma1, linewidth=3, color="#fff4eb")
plt.plot(ma2, linewidth=3, color="#fff4eb")
plt.plot(ma3, linewidth=3, color="#fff4eb")
plt.plot(ma4, linewidth=3, color="#fff4eb")
plt.plot(ma5, linewidth=3, color="#fff4eb")
plt.plot(ma6, linewidth=3, color="#fff4eb")
plt.plot(ma7, linewidth=3, color="#fff4eb")
plt.plot(ma8, linewidth=3, color="#fff4eb")
plt.plot(ma9, linewidth=3, color="#fff4eb")
plt.plot(ma10, linewidth=3, color="#fff4eb", label='Moving Reward')

plt.plot(mean_results, linewidth=4, color="#0595b2", label='Average Reward of 10 Different Seeds')
plt.plot(mean_ma_results, linewidth=4, color="#fb8402", label='Moving Average Reward of 10 Different Seeds')
plt.legend(prop={"family": "Times New Roman", "size": 24})
plt.savefig(path)
plt.show()
