# HACC_GPM

![build](https://github.com/humz2k/HACC_GPM/actions/workflows/helloAction.yml/badge.svg)

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
#### Building HACC_GPM
```
make
```

`haccgpm` needs `python3` with `camb` and `numpy` installed. To link to a specific version of python, build with `make PY_LIB=-lpython3.xx[m]`.

To build without python, use `make nopython`.

[Docs](https://humz2k.github.io/HACC_GPM-Docs/)