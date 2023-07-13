# HACC_GPM

### Building From Source
These are the steps to build from source:
#### Cloning Repo
```
git clone https://github.com/humz2k/HACC_GPM.git
cd HACC_GPM
```
#### Init Submodules
```
git submodule init
git submodule update
```
#### Building GC
```
cd bdwgc
git clone https://github.com/ivmai/libatomic_ops.git
make -f Makefile.direct check
cd ..
```
#### Building HACC_GPM
```
make main
```

Needs `python3` with `camb` and `numpy` installed. To link to a specific version of python, build with `make main PY_LIB=-lpython3.xx[m]`.

[Docs](https://humz2k.github.io/HACC_GPM-Docs/)