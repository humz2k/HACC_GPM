PY_C_FLAGS := $(shell python3-config --cflags)
PY_LD_FLAGS := $(shell python3-config --ldflags)
PY_NP_FLAGS := $(shell python3 -c "import numpy as np;print(np.get_include())")

SWFFT_DIR := swfft-all-to-all
PYCOSMO_DIR := pycosmotools

HACCGPM_BUILD_DIR ?= build

HACCGPM_NOPYTHON_DIR ?= build/nopython

CUDA_DIR ?= /usr/local/cuda

CUDA_ARCH_FLAGS ?= -arch=sm_60 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86

PY_LIB ?= -lpython3.9

HACCGPM_INCLUDE ?= -Isrc -Ipycosmotools/include

HACCGPM_BUILD_FOLDERS ?=

include src/*/*.include

all: main nopython

thing1:
	echo $(HACCGPM_FILES)
	echo $(HACCGPM_NOPYTHON_FILES)
#echo $(HACCGPM_CPP_FILES)
	

main: $(HACCGPM_BUILD_DIR)/driver_pm.o $(HACCGPM_FILES) $(HACCGPM_BUILD_DIR)/ccamb.o | swfft pycosmo
	mpicxx $^ $(SWFFT_DIR)/lib/swfft_a2a_gpu.a -L$(CUDA_DIR)/lib64 -lcudart -lcufft $(PY_LD_FLAGS) $(PY_LIB) -L$(PYCOSMO_DIR)/lib -lpycosmo -I$(CUDA_DIR)/include $(HACCGPM_INCLUDE) -fPIC -O3 -fopenmp -g -o haccgpm

nopython: $(HACCGPM_NOPYTHON_DIR)/driver_pm.o $(HACCGPM_NOPYTHON_FILES) | swfft
	mpicxx $^ -DNOPYTHON $(SWFFT_DIR)/lib/swfft_a2a_gpu.a -L$(CUDA_DIR)/lib64 -lcudart -lcufft -I$(CUDA_DIR)/include $(HACCGPM_INCLUDE) -fPIC -O3 -fopenmp -g -o haccgpmnopython

swfft:
	cd $(SWFFT_DIR) && $(MAKE) alltoallgpu

pycosmo:
	cd $(PYCOSMO_DIR) && $(MAKE) PY_LIB=$(PY_LIB)

$(HACCGPM_BUILD_DIR):
	mkdir -p $(HACCGPM_BUILD_DIR)
	$(foreach dir,$(HACCGPM_BUILD_FOLDERS),mkdir -p $(dir);)
#	mkdir -p $(HACCGPM_BUILD_DIR)/pm_solver
#	mkdir -p $(HACCGPM_BUILD_DIR)/pm_solver/cuda

$(HACCGPM_NOPYTHON_DIR): 
	mkdir -p $(HACCGPM_NOPYTHON_DIR)

$(HACCGPM_BUILD_DIR)/ccamb.o: cambTools/ccamb.c | $(HACCGPM_BUILD_DIR)
	python3 cambTools/package_cambpy.py
	gcc $< $(PY_C_FLAGS) -o $@ $(PY_LD_FLAGS) $(PY_LIB) -I$(PY_NP_FLAGS) -fPIC -c -O3 -Wno-unused-but-set-variable -Wno-return-type

$(HACCGPM_BUILD_DIR)/%.o: */%.cpp | $(HACCGPM_BUILD_DIR)
	mpicxx $< -I$(CUDA_DIR)/include $(HACCGPM_INCLUDE) -fPIC -O3 -fopenmp -g -c -o $@ -Wall

$(HACCGPM_BUILD_DIR)/%.o: */%.cu | $(HACCGPM_BUILD_DIR)
	nvcc $< -lcufft -lineinfo -Xptxas -v -Xcompiler -fPIC,-O3,-fopenmp,-g,-Wall, $(CUDA_ARCH_FLAGS) $(HACCGPM_INCLUDE) -c -o $@ 

$(HACCGPM_BUILD_DIR)/%.o: */*/%.cpp | $(HACCGPM_BUILD_DIR)
	mpicxx $< -I$(CUDA_DIR)/include $(HACCGPM_INCLUDE) -fPIC -O3 -fopenmp -g -c -o $@ -Wall

$(HACCGPM_BUILD_DIR)/%.o: */*/%.cu | $(HACCGPM_BUILD_DIR)
	nvcc $< -lcufft -lineinfo -Xptxas -v -Xcompiler -fPIC,-O3,-fopenmp,-g,-Wall, $(CUDA_ARCH_FLAGS) $(HACCGPM_INCLUDE) -c -o $@ 

$(HACCGPM_BUILD_DIR)/%.o: */*/*/%.cpp | $(HACCGPM_BUILD_DIR)
	mpicxx $< -I$(CUDA_DIR)/include $(HACCGPM_INCLUDE) -fPIC -O3 -fopenmp -g -c -o $@ -Wall

$(HACCGPM_BUILD_DIR)/%.o: */*/*/%.cu | $(HACCGPM_BUILD_DIR)
	nvcc $< -lcufft -lineinfo -Xptxas -v -Xcompiler -fPIC,-O3,-fopenmp,-g,-Wall, $(CUDA_ARCH_FLAGS) $(HACCGPM_INCLUDE) -c -o $@ 

$(HACCGPM_BUILD_DIR)/%.o: */*/*/*/%.cpp | $(HACCGPM_BUILD_DIR)
	mpicxx $< -I$(CUDA_DIR)/include $(HACCGPM_INCLUDE) -fPIC -O3 -fopenmp -g -c -o $@ -Wall

$(HACCGPM_BUILD_DIR)/%.o: */*/*/*/%.cu | $(HACCGPM_BUILD_DIR)
	nvcc $< -lcufft -lineinfo -Xptxas -v -Xcompiler -fPIC,-O3,-fopenmp,-g,-Wall, $(CUDA_ARCH_FLAGS) $(HACCGPM_INCLUDE) -c -o $@ 

$(HACCGPM_NOPYTHON_DIR)/%.o: */%.cpp | $(HACCGPM_BUILD_DIR) $(HACCGPM_NOPYTHON_DIR)
	mpicxx $< -DNOPYTHON -I$(CUDA_DIR)/include $(HACCGPM_INCLUDE) -fPIC -O3 -fopenmp -g -c -o $@ -Wall

$(HACCGPM_NOPYTHON_DIR)/%.o: */%.cu | $(HACCGPM_BUILD_DIR) $(HACCGPM_NOPYTHON_DIR)
	nvcc $< -DNOPYTHON -lcufft -lineinfo -Xptxas -v -Xcompiler -fPIC,-O3,-fopenmp,-g,-Wall, $(CUDA_ARCH_FLAGS) $(HACCGPM_INCLUDE) -c -o $@ 

$(HACCGPM_NOPYTHON_DIR)/%.o: */*/%.cpp | $(HACCGPM_BUILD_DIR) $(HACCGPM_NOPYTHON_DIR)
	mpicxx $< -DNOPYTHON -I$(CUDA_DIR)/include $(HACCGPM_INCLUDE) -fPIC -O3 -fopenmp -g -c -o $@ -Wall

$(HACCGPM_NOPYTHON_DIR)/%.o: */*/%.cu | $(HACCGPM_BUILD_DIR) $(HACCGPM_NOPYTHON_DIR)
	nvcc $< -DNOPYTHON -lcufft -lineinfo -Xptxas -v -Xcompiler -fPIC,-O3,-fopenmp,-g,-Wall, $(CUDA_ARCH_FLAGS) $(HACCGPM_INCLUDE) -c -o $@ 

.PHONY: clean
clean:
	rm -f haccgpm
	rm -f haccgpmnopython
	rm -rf $(HACCGPM_BUILD_DIR)
	cd $(SWFFT_DIR) && $(MAKE) clean
