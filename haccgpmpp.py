#!/home/hqureshi/miniconda3/bin/python

import postprocessing as pp
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":

    pp.plot_pretty(dpi=300)

    run = pp.Run("runs/test")

    to_plot = [int(i) for i in sys.argv[2:]]

    for i in to_plot:
        pk = run.pks[i]

        pk.plot(folds=[0],label="z=" + str(round(pk.z,1)) + ", fold=0",linewidth=0.9)
        if (len(pk.folds) > 1):
            pk.plot(folds=[1],label="z=" + str(round(pk.z,1)) + ", fold=1",linewidth=0.9)

    run.plot_k_nyq(color="black",linestyles="dashed",linewidth=0.5)
    run.plot_k_nyq_2(color="black",linestyles="dashed",linewidth=0.5)
    plt.legend()

    plt.title("Power Spectrum\n" + r"$N_g = " + str(run.params["NG"]) + r"$, $N_p = " + str(run.params["NP"]) + r"$, $L = " + str(run.params["RL"]) + r"$ Mpc",size=10)

    plt.tight_layout()
    plt.savefig(sys.argv[1])