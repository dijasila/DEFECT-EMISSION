import os 
import sys
import json
import pickle
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt


sys.path.append("/home/niflheim/markas/TEMPO/defect-emission/tensor_networks/classical_tempo_cavity")

import tempo_class as tempo





system = pickle.load(open("results.pckl", "rb"))


fig, ax = plt.subplots(ncols=2)
ax[0].plot(system.trange, system.expect_data[0], '.')


ax[1].plot(system.trange, system.expect_data[0], '.')


for a in ax:
    pass
    # a.set_xlim(0,system.trange[-1])
    # a.set_ylim(0,1)


#plt.plot(sys.trange, np.exp(reorg) * corr_func.real)
plt.show()

