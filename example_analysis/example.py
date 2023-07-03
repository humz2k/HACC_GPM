import numpy as np
import matplotlib.pyplot as plt

maxs = []
steps = []
means = []

initial = None

def onAnalysisStep(**kwargs):
    global initial,maxs,means,steps
    #print("Python Analysis Step")
    step = kwargs["step"]
    z = kwargs["z"]
    a = kwargs["a"]
    particles = kwargs["particles"]
    ng = kwargs["ng"]
    rl = kwargs["rl"]

    if (type(initial) == type(None)):
        raw = np.arange(ng*ng*ng)
        xs = (raw//ng)//ng
        ys = (raw - ng*ng*xs)//ng
        zs = raw - ng*ng*xs - ng*ys
        raw = np.column_stack((xs,ys,zs))
        initial = raw

    diff = np.abs(particles[:,[0,1,2]] - initial)
    diff[diff > (ng/2)] -= ng
    maxs.append(np.max(np.abs(diff)))
    means.append(np.mean(np.abs(diff)))

    steps.append(step)
    if (step == 624):
        pass
        print(maxs)
        print(means)
        plt.plot(steps,maxs,label="max")
        plt.plot(steps,means,label="mean")
        plt.ylabel("Delta")
        plt.xlabel("step")
        plt.title("Delta From Initial Position")
        plt.legend()
        plt.tight_layout()
        plt.savefig("max_delta.jpg")
        plt.close()