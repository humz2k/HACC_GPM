PY_C_FLAGS := $(shell python3-config --cflags)
PY_LD_FLAGS := $(shell python3-config --ldflags)
PY_NP_FLAGS := $(shell python3 -c "import numpy as np;print(np.get_include())")

RECAP_BUILD_DIR ?= build

main: driver.cu $(RECAP_BUILD_DIR)/ccamb.o $(RECAP_BUILD_DIR)/cic.o $(RECAP_BUILD_DIR)/initializer.o $(RECAP_BUILD_DIR)/io.o $(RECAP_BUILD_DIR)/power.o $(RECAP_BUILD_DIR)/greens.o $(RECAP_BUILD_DIR)/solver.o $(RECAP_BUILD_DIR)/params.o $(RECAP_BUILD_DIR)/timestepper.o $(RECAP_BUILD_DIR)/mmanager.o $(RECAP_BUILD_DIR)/ffts.o
	nvcc $^ -lcufft -lineinfo -Xptxas -v $(PY_LD_FLAGS) -lpython3.9 -Xcompiler -fPIC,-O3,-fopenmp,-g, -gencode arch=compute_60,code=sm_60 -o haccgpm

$(RECAP_BUILD_DIR): 
	mkdir -p $(RECAP_BUILD_DIR)

$(RECAP_BUILD_DIR)/ccamb.o: cambTools/ccamb.c | $(RECAP_BUILD_DIR)
	python3 cambTools/package_cambpy.py
	gcc $< $(PY_C_FLAGS) -o $@ $(PY_LD_FLAGS) -lpython3.9 -I$(PY_NP_FLAGS) -c -O3 -Wno-unused-but-set-variable -Wno-return-type

$(RECAP_BUILD_DIR)/%.o: %.cu | $(RECAP_BUILD_DIR)
	nvcc $< -lcufft -lineinfo -Xptxas -v -Xcompiler -fPIC,-O3,-fopenmp,-g, -gencode arch=compute_60,code=sm_60 -c -o $@

.PHONY: clean
clean:
	rm -f haccgpm
	rm -rf $(RECAP_BUILD_DIR)
