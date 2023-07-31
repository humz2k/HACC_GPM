#!/home/hqureshi/miniconda3/bin/python

class MemoryCalculator:
    def __init__(self,use_single_fft,use_single_greens,use_float3,use_one_grid,use_temp_grid,use_greens_cache):
        self.use_single_fft = use_single_fft
        self.use_single_greens = use_single_greens
        self.use_float3 = use_float3
        self.use_one_grid = use_one_grid
        self.use_temp_grid = use_temp_grid
        self.use_greens_cache = use_greens_cache
    
    