import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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


plt.figure()
plt.title("seed_rewards_curve")
plt.xlabel('episodes')
plt.ylabel('Reward of 10 Different Seeds')
plt.plot(seed1, label='Seed=1')      # 奖励
plt.plot(seed2, label='Seed=2')
plt.plot(seed3, label='Seed=3')
plt.plot(seed4, label='Seed=4')
plt.plot(seed5, label='Seed=5')
plt.plot(seed6, label='Seed=6')
plt.plot(seed7, label='Seed=7')
plt.plot(seed8, label='Seed=8')
plt.plot(seed9, label='Seed=9')
plt.plot(seed10, label='Seed=10')

plt.legend()
plt.savefig("../code/outputs/seed/seed_results/seed_rewards_curve")
plt.show()
