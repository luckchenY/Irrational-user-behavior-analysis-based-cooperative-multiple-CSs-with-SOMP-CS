import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

seed1 = np.load("../outputs/Policy_Gradient/seed1/results/train_rewards.npy")
seed2 = np.load("../outputs/Policy_Gradient/seed1/results/train_rewards.npy")
seed3 = np.load("../outputs/Policy_Gradient/seed1/results/train_rewards.npy")
seed4 = np.load("../outputs/Policy_Gradient/seed1/results/train_rewards.npy")
seed5 = np.load("../outputs/Policy_Gradient/seed1/results/train_rewards.npy")
seed6 = np.load("../outputs/Policy_Gradient/seed1/results/train_rewards.npy")
seed7 = np.load("../outputs/Policy_Gradient/seed1/results/train_rewards.npy")
seed8 = np.load("../outputs/Policy_Gradient/seed1/results/train_rewards.npy")
seed9 = np.load("../outputs/Policy_Gradient/seed1/results/train_rewards.npy")
seed10 = np.load("../outputs/Policy_Gradient/seed1/results/train_rewards.npy")

ma1 = np.load("../outputs/Policy_Gradient/seed1/results/train_ma_rewards.npy")
ma2 = np.load("../outputs/Policy_Gradient/seed1/results/train_ma_rewards.npy")
ma3 = np.load("../outputs/Policy_Gradient/seed1/results/train_ma_rewards.npy")
ma4 = np.load("../outputs/Policy_Gradient/seed1/results/train_ma_rewards.npy")
ma5 = np.load("../outputs/Policy_Gradient/seed1/results/train_ma_rewards.npy")
ma6 = np.load("../outputs/Policy_Gradient/seed1/results/train_ma_rewards.npy")
ma7 = np.load("../outputs/Policy_Gradient/seed1/results/train_ma_rewards.npy")
ma8 = np.load("../outputs/Policy_Gradient/seed1/results/train_ma_rewards.npy")
ma9 = np.load("../outputs/Policy_Gradient/seed1/results/train_ma_rewards.npy")
ma10 = np.load("../outputs/Policy_Gradient/seed1/results/train_ma_rewards.npy")

mean_results = np.zeros(1000)
mean_ma_results = np.zeros(1000)

for j in range(1000):
    mean_results[j] = (seed1[j]+seed2[j]+seed3[j]+seed4[j]+seed5[j]+seed6[j]+seed7[j]+seed8[j]+seed9[j]+seed10[j])/10
    mean_ma_results[j] = (ma1[j]+ma2[j]+ma3[j]+ma4[j]+ma5[j]+ma6[j]+ma7[j]+ma8[j]+ma9[j]+ma10[j])/10

path = "../outputs/Policy_Gradient/mean_results"
np.save(path+'mean_results.npy', mean_results)
np.save(path+'mean_ma_results.npy', mean_ma_results)

plt.figure()
plt.title("mean_rewards_curve")
plt.xlabel('Episodes')
plt.ylabel('Average Reward of 10 Different Seeds')
plt.plot(mean_results, label='Average Reward of 10 Different Seeds')
plt.plot(mean_ma_results, label='Moving Average Reward of 10 Different Seeds')
plt.legend()
plt.savefig(path+"mean_rewards_curve")
plt.show()