import numpy as np

with open('dat.txt',"r") as f:
    dat = f.read().splitlines()

maybe = np.array([int(i.split("=")[0]) for i in dat])
actual = np.array([int(i.split(")")[1].split(" ")[-1]) for i in dat])
print(np.allclose(maybe,actual))