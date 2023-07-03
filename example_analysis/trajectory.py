import numpy as np
import matplotlib.pyplot as plt

ps = [500,499,498,497,496,495]
xs = [[] for i in ps]
ys = [[] for i in ps]
zs = [[] for i in ps]

def onAnalysisStep(**kwargs):
    global ps,xs,ys,zs
    step = kwargs["step"]
    z = kwargs["z"]
    a = kwargs["a"]
    particles = kwargs["particles"]
    ng = kwargs["ng"]
    rl = kwargs["rl"]

    for idx,p in enumerate(ps):
        xs[idx].append(particles[p,0])
        ys[idx].append(particles[p,1])
        zs[idx].append(particles[p,2])

    if (step == 624):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for idx,p in enumerate(ps):
            ax.plot(xs[idx],ys[idx],zs[idx],label="id=" + str(p))
        plt.legend()
        plt.tight_layout()
        plt.savefig("trajectory.jpg")
        plt.close()

