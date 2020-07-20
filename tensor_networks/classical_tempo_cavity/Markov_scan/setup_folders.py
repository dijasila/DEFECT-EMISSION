import os 
import json

kappa = 10


gs = [0.1, 1, 5, 10, 20, 30]
mem_cuts = [1, 5, 10, 30, 50, 70, 100]



for g in gs:
    for mem_cut in mem_cuts:
        data_path = "g_{}/mem_{}/".format(g, mem_cut)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        params = {}
        params["g"] = g
        params["mem_cut"] = mem_cut
        params["kappa"] = kappa
        
        with open(data_path + "params.json", "w") as f:
            json.dump(params, f)

        os.system("cp cavity_run.py {}".format(data_path))
        
        cwd = os.getcwd()
        print(cwd)
        os.chdir(cwd + "/" + data_path)
        os.system("mq submit cavity_run.py -R 8:5h")
        os.chdir(cwd)
