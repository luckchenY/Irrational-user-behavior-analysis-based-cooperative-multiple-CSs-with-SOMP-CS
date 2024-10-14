import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

grad_dqn = np.load("../code/outputs/gradient_comparison/dqn/results/gradient.npy")
print("grad_dqn", grad_dqn)

grad_gd = np.load("../code/outputs/gradient_comparison/pg/results/gradient.npy")
print("grad_gd", grad_gd)

path = "../code/outputs/gradient_comparison/dqn_pg"

plt.figure(figsize=(12, 9))
plt.title("")
plt.xticks(fontname="Times New Roman", fontsize=26)
plt.yticks(fontname="Times New Roman", fontsize=26)
plt.xlabel('Layer', fontdict={"family": "Times New Roman", "size": 32})
plt.ylabel('Gradient of Convergence ', fontdict={"family": "Times New Roman", "size": 32})
plt.plot(grad_dqn, linewidth=3, color="#cf030c", label='Deep Q-Learning')
plt.plot(grad_gd, linewidth=3, color="#0c365e", label='Policy Gradient')


plt.legend(prop={"family": "Times New Roman", "size": 24})
plt.savefig(path)
plt.show()

sys.exit()
