PY_C_FLAGS := $(shell python3-config --cflags)
PY_LD_FLAGS := $(shell python3-config --ldflags)
PY_NP_FLAGS := $(shell python3 -c "import numpy as np;print(np.get_include())")

RECAP_BUILD_DIR ?= build

main: driver.cu $(RECAP_BUILD_DIR)/ccamb.o $(RECAP_BUILD_DIR)/cic.o $(RECAP_BUILD_DIR)/initializer.o $(RECAP_BUILD_DIR)/io.o $(RECAP_BUILD_DIR)/power.o $(RECAP_BUILD_DIR)/greens.o $(RECAP_BUILD_DIR)/solver.o $(RECAP_BUILD_DIR)/params.o $(RECAP_BUILD_DIR)/timestepper.o $(RECAP_BUILD_DIR)/mmanager.o $(RECAP_BUILD_DIR)/ffts.o
	nvcc $^ -lcufft -lineinfo -Xptxas -v $(PY_LD_FLAGS) -lpython3.9 -Xcompiler -fPIC,-O3,-fopenmp,-g, -gencode arch=compute_60,code=sm_60 -o haccgpm

$(RECAP_BUILD_DIR)/ccamb.o: cambTools/ccamb.c
	gcc $< $(PY_C_FLAGS) -o $@ $(PY_LD_FLAGS) -lpython3.9 -I$(PY_NP_FLAGS) -c -O3 -Wno-unused-but-set-variable -Wno-return-type

$(RECAP_BUILD_DIR)/%.o: %.cu
	nvcc $< -lcufft -lineinfo -Xptxas -v -Xcompiler -fPIC,-O3,-fopenmp,-g, -gencode arch=compute_60,code=sm_60 -c -o $@

.PHONY: clean
clean:
	rm -f haccgpm
	rm -f $(RECAP_BUILD_DIR)/ccamb.o
	rm -f $(RECAP_BUILD_DIR)/cic.o
	rm -f $(RECAP_BUILD_DIR)/initializer.o
	rm -f $(RECAP_BUILD_DIR)/io.o
	rm -f $(RECAP_BUILD_DIR)/power.o
	rm -f $(RECAP_BUILD_DIR)/greens.o
	rm -f $(RECAP_BUILD_DIR)/solver.o
	rm -f $(RECAP_BUILD_DIR)/params.o
	rm -f $(RECAP_BUILD_DIR)/timestepper.o
	rm -f $(RECAP_BUILD_DIR)/mmanager.o
	rm -f $(RECAP_BUILD_DIR)/ffts.o
