import numpy as np
import matplotlib.pyplot as plt

def onAnalysisStep(**kwargs):

    step = kwargs["step"]
    z = kwargs["z"]
    a = kwargs["a"]
    particles = kwargs["particles"]
    ng = kwargs["ng"]
    rl = kwargs["rl"]

    center = np.array([ng/2,ng/2,ng/2])
    radius = 5

    particles = particles[:,[0,1,2]]
    dists = np.linalg.norm(particles - center,axis=1)
    print(dists)
    particles = particles[dists < radius]
    particles -= center

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(particles[:,0],particles[:,1],particles[:,2])
    ax.set_xlim(-radius,radius)
    ax.set_ylim(-radius,radius)
    ax.set_zlim(-radius,radius)
    plt.tight_layout()
    plt.savefig("frames/step" + str(step) + ".jpg")
    plt.close()

