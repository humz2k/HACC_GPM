import numpy as np
import matplotlib.pyplot as plt

boundaries = []
steps = []
exits = []
returns = []

global_ranks = None
global_on_boundary = None

def onAnalysisStep(**kwargs):
    global boundaries,steps,global_ranks,global_on_boundary,exits,returns
    step = kwargs["step"]
    z = kwargs["z"]
    a = kwargs["a"]
    particles = kwargs["particles"]
    ng = kwargs["ng"]
    rl = kwargs["rl"]

    ranks = (particles[:,[0,1,2]]//(ng//2))
    rank_centers = ranks*(ng//2)
    rank_coords = particles[:,[0,1,2]] - rank_centers
    on_boundary = rank_coords > ((ng//2) - 1)
    on_boundary = np.logical_or(np.logical_or(on_boundary[:,0],on_boundary[:,1]),on_boundary[:,2])
    ranks = ranks[:,0]*4 + ranks[:,1]*2 + ranks[:,2]
    if step != 0:
        exitted = np.logical_and(on_boundary,np.logical_not(global_on_boundary))
        entered = np.logical_and(np.logical_not(on_boundary),global_on_boundary)
        previous_rank = global_ranks[entered]
        current_rank = ranks[entered]
        returned = previous_rank == current_rank
        exits.append(np.sum(exitted))
        returns.append(np.sum(returned))

    global_ranks = np.copy(ranks)
    global_on_boundary = np.copy(on_boundary)

    boundaries.append(np.sum(on_boundary))
    steps.append(step)

    with open("data.txt","w") as f:
        f.write("step,exit,return\n")
        for step,i,j in zip(steps[1:],exits,returns):
            f.write(str(step)+","+str(i)+","+str(j)+"\n")

    if (step == 624):
    #    plt.title("Particles on Boundary\nng=" + str(ng) + ", rl=" + str(rl))
        plt.plot(steps[1:],exits)
        plt.plot(steps[1:],returns)
        plt.xlabel("step")
        plt.savefig("tmp.jpg")
    #    plt.ylabel("# of particles on boundary")
    #    plt.tight_layout()
    #    plt.savefig('on_boundary_ng' + str(ng) + "_rl" + str(int(rl)) + '.jpg')
    