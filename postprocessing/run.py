import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def z2a(z):
    return 1/(z + 1)

def a2z(a):
    return (1/a) - 1

class PowerSpectrum:
    def __init__(self,z,step,folds,c):
        self.folds = folds
        self.step = step
        self.z = z
        self.c = c
    
    def cmap(self,cm):
        cmap = matplotlib.cm.get_cmap(cm)
        return cmap(self.c)

    def plot(self,folds=[0],**kwargs):
        plt.xscale('log')
        plt.yscale('log')
        for fold in folds:
            try:
                ks = self.folds[fold]["k"]
                pks = self.folds[fold]["P_0(k)"]
            except:
                print("Can't find fold " + str(fold))
                exit()
            plt.plot(ks,pks,**kwargs)
    
    def __str__(self):
        return "PowerSpectrum<z=" + str(self.z) + ",step=" + str(self.step) + ">"

class Run:
    def __init__(self,base_name):
        self.base_name = base_name
        self.params = self.read_params()
        self.pks = self.find_pks()
        self.k_nyq = (np.pi*self.params["NG"])/self.params["RL"]
    
    def plot_k_nyq(self,**kwargs):
        ymin,ymax = plt.ylim()
        plt.vlines([self.k_nyq],ymin,ymax,**kwargs)
        plt.ylim(ymin,ymax)
    
    def plot_k_nyq_2(self,**kwargs):
        ymin,ymax = plt.ylim()
        plt.vlines([self.k_nyq/2],ymin,ymax,**kwargs)
        plt.ylim(ymin,ymax)

    def plot_k_nyq_4(self,**kwargs):
        ymin,ymax = plt.ylim()
        plt.vlines([self.k_nyq/2],ymin,ymax,**kwargs)
        plt.ylim(ymin,ymax)

    def find_pks(self):
        a_init = z2a(self.params["Z_IN"])
        deltaT = (z2a(self.params["Z_FIN"]) - z2a(self.params["Z_IN"]))/self.params["N_STEPS"]
        out = []
        try:
            pks = [self.read_pk("ini",fold) for fold in range(self.params["PK_FOLDS"] + 1)]
            pk = PowerSpectrum(float(self.params["Z_IN"]),"ini",pks,0.0)
            out.append(pk)
        except:
            pass
        for i in range(self.params["N_STEPS"]):
            current_a = a_init + (deltaT * i)
            current_z = a2z(current_a)
            try:
                pks = [self.read_pk(i,fold) for fold in range(self.params["PK_FOLDS"] + 1)]
                pk = PowerSpectrum(current_z,i,pks,(i+1)/self.params["N_STEPS"])
                out.append(pk)
            except:
                pass
        last_step = self.params["N_STEPS"]
        if "LAST_STEP" in self.params:
            last_step = self.params["LAST_STEP"]
        current_a = a_init + (deltaT * last_step)
        current_z = a2z(current_a)
        try:
            pks = [self.read_pk("fin",fold) for fold in range(self.params["PK_FOLDS"] + 1)]
            pk = PowerSpectrum(current_z,"fin",pks,1.0)
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

def plot_pretty(dpi=150,fontsize=15):
    plt.rcParams['figure.dpi']= dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    plt.rc('legend',fontsize=5)
    #plt.rc('text', usetex=True)
    plt.rcParams['figure.figsize'] = [5, 4]

    