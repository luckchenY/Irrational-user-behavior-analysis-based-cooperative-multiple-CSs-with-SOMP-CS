from mimetypes import read_mime_types
import os
from turtle import width
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import seaborn as sns

EVLink_Price = np.load("../outputs/EV_Charging_Pricing/example/results/EVLink_Prices.npy")
prices1 = np.load("../outputs/EV_Charging_Pricing/example/results/prices1.npy")
prices2 = np.load("../outputs/EV_Charging_Pricing/example/results/prices2.npy")
rates1 = np.load("../outputs/EV_Charging_Pricing/example/results/rates1.npy")
rates2 = np.load("../outputs/EV_Charging_Pricing/example/results/rates2.npy")

EVLink_Price=EVLink_Price[10:]
prices1=prices1[10:]
prices2=prices2[10:]
rates1=rates1[10:]
rates2=rates2[10:]

path1 = "../outputs/EV_Charging_Pricing/example/pricing_example"
plt.figure(figsize=(12, 9))
plt.title("")
plt.xticks(fontname="Times New Roman", fontsize=26)
plt.yticks(fontname="Times New Roman", fontsize=26)
plt.xlabel('Time', fontdict={"family": "Times New Roman", "size": 32})
plt.ylabel('Price (CNY)', fontdict={"family": "Times New Roman", "size": 32})
plt.plot(EVLink_Price, linewidth=3, color="#244653", label='EVLink Price')
plt.plot(prices1, linewidth=3, color="#cc3e00", label='Price of Station 1')
plt.plot(prices2, linewidth=3, color="#ff9900", label='Price of Station 2')
plt.legend(loc="lower right", prop={"family": "Times New Roman", "size": 24})
plt.savefig(path1)
plt.show()

path2 = "../outputs/EV_Charging_Pricing/example/charging_rate_example"
plt.figure(figsize=(12, 9))
plt.title("")
plt.xticks(fontname="Times New Roman", fontsize=26)
plt.yticks(fontname="Times New Roman", fontsize=26)
plt.xlabel('Time', fontdict={"family": "Times New Roman", "size": 32})
plt.ylabel('Charging Rate (KWh)', fontdict={"family": "Times New Roman", "size": 32})
plt.plot(EVLink_Price, linewidth=3, color="#244653", label='EVLink Price')
plt.plot(rates1, linewidth=3, color="#cc3e00", label='Rate of Station 1')
plt.plot(rates2, linewidth=3, color="#ff9900", label='Rate of Station 2')
plt.legend(loc="lower right", prop={"family": "Times New Roman", "size": 24})
plt.savefig(path2)
plt.show()


sys.exit()
