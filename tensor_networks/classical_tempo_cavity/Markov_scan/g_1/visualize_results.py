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




mem_cuts = [1, 5, 10, 30, 50, 70, 100]

fig, ax = plt.subplots(ncols = 2)

for mem_cut in mem_cuts:
    data_path = "mem_{}/".format(mem_cut)
    system = pickle.load(open(data_path + "results.pckl", "rb"))
    
    
    ax[0].plot(system.trange, system.expect_data[0], '.', label = "{}".format(mem_cut))
    ax[1].plot(system.trange, system.expect_data[0], '.')
    ax[0].set_xlim([0, 3])
    ax[1].set_xlim([0, 3])


ax[0].legend()
plt.show()
