import numpy as np
import matplotlib.pyplot as plt

boundaries = []
steps = []

def onAnalysisStep(**kwargs):
    global boundaries,steps
    step = kwargs["step"]
    z = kwargs["z"]
    a = kwargs["a"]
    particles = kwargs["particles"]
    ng = kwargs["ng"]
    rl = kwargs["rl"]

    rank_centers = (particles[:,[0,1,2]]//(ng//2))*(ng//2)
    rank_coords = particles[:,[0,1,2]] - rank_centers
    on_boundary = rank_coords > ((ng//2) - 1)
    on_boundary = np.logical_or(np.logical_or(on_boundary[:,0],on_boundary[:,1]),on_boundary[:,2])

    boundaries.append(np.sum(on_boundary))
    steps.append(step)

    if (step == 624):
        plt.title("Particles on Boundary\nng=" + str(ng) + ", rl=" + str(rl))
        plt.plot(steps,boundaries)
        plt.xlabel("step")
        plt.ylabel("# of particles on boundary")
        plt.tight_layout()
        plt.savefig('on_boundary_ng' + str(ng) + "_rl" + str(int(rl)) + '.jpg')
    