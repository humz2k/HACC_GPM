import os

path = os.path.dirname(__file__)

#print(str(Path.cwd()) + "/cambpy.py")

with open(os.path.join(path,"cambpy.py"),"r") as f:
    raw = f.read()

raw = '#define CAMBPYSTR "' + raw.replace("\n",r"\n") + '"'

with open(os.path.join(path,"cambpy.h"),"w") as f:
    f.write(raw)