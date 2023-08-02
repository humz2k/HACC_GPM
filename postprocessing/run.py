import numpy as np
import pandas as pd

def z2a(z):
    return 1/(z + 1)

def a2z(a):
    return (1/a) - 1

class PowerSpectrum:
    def __init__(self,z,step,folds):
        self.folds = folds
        self.step = step
        self.z = z
    
    def __str__(self):
        return "PowerSpectrum<z=" + str(self.z) + ",step=" + str(self.step) + ">"

class Run:
    def __init__(self,base_name):
        self.base_name = base_name
        self.params = self.read_params()
        self.pks = self.find_pks()

    def find_pks(self):
        a_init = z2a(self.params["Z_IN"])
        deltaT = (z2a(self.params["Z_FIN"]) - z2a(self.params["Z_IN"]))/self.params["N_STEPS"]
        out = []
        for i in range(self.params["N_STEPS"]):
            current_a = a_init + (deltaT * i)
            current_z = a2z(current_a)
            try:
                pks = [self.read_pk(i,fold) for fold in range(self.params["PK_FOLDS"] + 1)]
                pk = PowerSpectrum(current_z,i,pks)
                out.append(pk)
            except:
                pass
        return out
    
    def read_params(self):
        fname = self.base_name + ".params"
        with open(fname,"r") as f:
            raw = [i.split(" ") for i in f.read().splitlines()]
        data = {}
        for i,j in raw:
            try:
                data[i] = int(j)
            except:
                try:
                    data[i] = float(j)
                except:
                    data[i] = j
        return data

    def read_pk(self,step,fold):
        fname = self.base_name + ".pk." + str(step) + ".fold." + str(fold)
        with open(fname,"r") as f:
            raw = np.array([[float(j) for j in i.split(",")] for i in f.read().splitlines()])
        out = pd.DataFrame(raw,columns=["k","P_0(k)","nmodes"])
        out["nmodes"] = out["nmodes"].astype('int')
        return out
    
run = Run("/home/hqureshi/HACC_GPM/runs/test")

    