PY_C_FLAGS := $(shell python3-config --cflags)
PY_LD_FLAGS := $(shell python3-config --ldflags)
PY_NP_FLAGS := $(shell python3 -c "import numpy as np;print(np.get_include())")

SWFFT_DIR := swfft-all-to-all
BDWGC_DIR := bdwgc
PYCOSMO_DIR := pycosmotools

HACCGPM_BUILD_DIR ?= build

CUDA_DIR ?= /usr/local/cuda

CUDA_ARCH_FLAGS ?= -arch=sm_60 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86

PY_LIB ?= -lpython3.9

main: src/driver.cpp $(HACCGPM_BUILD_DIR)/swfftmanager.o $(HACCGPM_BUILD_DIR)/transfers.o $(HACCGPM_BUILD_DIR)/timers.o $(HACCGPM_BUILD_DIR)/ccamb.o $(HACCGPM_BUILD_DIR)/cic.o $(HACCGPM_BUILD_DIR)/initializer.o $(HACCGPM_BUILD_DIR)/io.o $(HACCGPM_BUILD_DIR)/power.o $(HACCGPM_BUILD_DIR)/greens.o $(HACCGPM_BUILD_DIR)/solver.o $(HACCGPM_BUILD_DIR)/params.o $(HACCGPM_BUILD_DIR)/timestepper.o $(HACCGPM_BUILD_DIR)/mmanager.o $(HACCGPM_BUILD_DIR)/ffts.o $(HACCGPM_BUILD_DIR)/particleswapkernels.o $(HACCGPM_BUILD_DIR)/cosmo.o | swfft pycosmo
	mpicxx $^ $(SWFFT_DIR)/lib/swfft_a2a_gpu.a bdwgc/libgc.a -L$(CUDA_DIR)/lib64 -lcudart -lcufft $(PY_LD_FLAGS) $(PY_LIB) -L$(PYCOSMO_DIR)/lib -lpycosmo -I$(CUDA_DIR)/include -fPIC -O3 -fopenmp -g -o haccgpm

swfft:
	cd $(SWFFT_DIR) && $(MAKE)

pycosmo:
	cd $(PYCOSMO_DIR) && $(MAKE) PY_LIB=$(PY_LIB)

bdwgc:
	cd $(BDWGC_DIR) && $(MAKE) -f Makefile.direct check

$(HACCGPM_BUILD_DIR): 
	mkdir -p $(HACCGPM_BUILD_DIR)

$(HACCGPM_BUILD_DIR)/ccamb.o: cambTools/ccamb.c | $(HACCGPM_BUILD_DIR)
	python3 cambTools/package_cambpy.py
	gcc $< $(PY_C_FLAGS) -o $@ $(PY_LD_FLAGS) $(PY_LIB) -I$(PY_NP_FLAGS) -fPIC -c -O3 -Wno-unused-but-set-variable -Wno-return-type

$(HACCGPM_BUILD_DIR)/%.o: src/%.cpp | $(HACCGPM_BUILD_DIR)
	mpicxx $< -I$(CUDA_DIR)/include -fPIC -O3 -fopenmp -g -c -o $@

$(HACCGPM_BUILD_DIR)/%.o: src/%.cu | $(HACCGPM_BUILD_DIR)
	nvcc $< -lcufft -lineinfo -Xptxas -v -Xcompiler -fPIC,-O3,-fopenmp,-g, $(CUDA_ARCH_FLAGS) -c -o $@

.PHONY: clean
clean:
	rm -f haccgpm
	rm -rf $(HACCGPM_BUILD_DIR)
# cd $(SWFFT_DIR) && $(MAKE) clean
