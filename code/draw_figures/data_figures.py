import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import seaborn as sns

evlink_price = np.load("../outputs/EV_Charging_Pricing/EV_LINK_Price/results/EVLink.npy")
# num_EV = np.load("../outputs/EV_Charging_Pricing/num of EV/results/# of EVs.npy")

#path1 = "../outputs/EV_Charging_Pricing/num of EV/results/num of EVs_curve"
path2 = "../outputs/EV_Charging_Pricing/EV_LINK_Price/results/EVLink_curve"

sns.set(style="darkgrid")
plt.figure(figsize=(12, 9))
plt.title("")
plt.xticks(fontname="Times New Roman", fontsize=26)
plt.yticks(fontname="Times New Roman", fontsize=26)
plt.xlabel('Times', fontdict={"family": "Times New Roman", "size": 32})
#plt.ylabel('Number of EVs being Charged', fontdict={"family": "Times New Roman", "size": 32})
plt.ylabel('Price (CNY)', fontdict={"family": "Times New Roman", "size": 32})
plt.plot(evlink_price, linewidth=3, color="steelblue", label='EVLink_Price')
#x = np.arange(0, 50)
#plt.bar(x, num_EV, color="#87cefa", label="# of EVs")

plt.legend(prop={"family": "Times New Roman", "size": 24})
plt.savefig(path2)
plt.show()

sys.exit()
