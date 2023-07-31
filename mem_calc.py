#!/home/hqureshi/miniconda3/bin/python

from math import floor
import sys

class MemoryCalculator:
    def __init__(self,use_single_fft,use_single_greens,use_float3,use_one_grid,use_temp_grid,use_greens_cache):
        self.use_single_fft = use_single_fft
        self.use_single_greens = use_single_greens
        self.use_float3 = use_float3
        self.use_one_grid = use_one_grid
        self.use_temp_grid = use_temp_grid
        self.use_greens_cache = use_greens_cache

        self.float_size = 4 #bytes
        self.float4_size = self.float_size * 4
        self.float3_size = self.float_size * 3
        self.int_size = 4 #bytes
        self.double_size = 8 #bytes
        self.deviceFFT_size = 2 * self.double_size
        self.floatFFT_size = 2 * self.float_size

    def forward(self,ng,pk_bins=221):
        particle_t = self.float4_size
        if self.use_float3:
            particle_t = self.float3_size

        grid_t = self.deviceFFT_size
        if self.use_single_fft:
            grid_t = self.floatFFT_size

        greens_t = 0
        if self.use_greens_cache:
            greens_t = self.double_size
            if self.use_single_greens:
                greens_t = self.float_size
        
        n_cells = ng * ng * ng

        d_vel = n_cells * particle_t
        d_pos = n_cells * particle_t
        d_greens = n_cells * greens_t

        d_grid = n_cells * grid_t

        d_x = d_grid
        d_y = d_grid
        d_z = d_grid
        if self.use_one_grid:
            d_x = 0
            d_y = 0
            d_z = 0
        
        d_temp_grid = 0
        if self.use_temp_grid:
            d_temp_grid = n_cells * self.float_size
        
        d_grad = n_cells * self.float4_size

        d_binCounts = self.int_size * pk_bins
        d_binVals = self.double_size * pk_bins

        total = d_binVals + d_binCounts + d_grad + d_temp_grid + d_x + d_y + d_z + 2*d_grid + d_greens + d_pos + d_vel
        individual = {
            "d_pos": d_pos * 1e-9,
            "d_vel": d_vel * 1e-9,
            "d_greens": d_greens * 1e-9,
            "d_grid": d_grid * 1e-9,
            "d_x": d_x * 1e-9,
            "d_y": d_y * 1e-9,
            "d_z": d_z * 1e-9,
            "d_temp_grid": d_temp_grid * 1e-9,
            "d_grad": d_grad * 1e-9,
            "d_binCounts": d_binCounts * 1e-9,
            "d_binVals": d_binVals * 1e-9
        }
        return total * 1e-9, individual #GB
    
    def backward(self,size,pk_bins=221): #size is GB
        size *= 1e+9

        size -= self.int_size * pk_bins
        size -= self.double_size * pk_bins

        particle_t = self.float4_size
        if self.use_float3:
            particle_t = self.float3_size

        grid_t = self.deviceFFT_size
        if self.use_single_fft:
            grid_t = self.floatFFT_size

        greens_t = 0
        if self.use_greens_cache:
            greens_t = self.double_size
            if self.use_single_greens:
                greens_t = self.float_size
        
        mul = 0

        mul += particle_t
        mul += particle_t
        mul += greens_t

        mul += grid_t

        if not self.use_one_grid:
            mul += 3 * grid_t
        
        if self.use_temp_grid:
            mul += self.float_size
        
        mul += self.float4_size

        n_cells = size / mul

        ng = floor(n_cells ** (1/3))

        return (floor(ng))
    
    def __call__(self,ng=None,size=None):
        if (type(ng) != type(None)):
            return self.forward(ng)
        if (type(size) != type(None)):
            return self.backward(size)
    
calc = MemoryCalculator(use_single_fft=True,
                        use_single_greens=True,
                        use_float3=True,
                        use_one_grid=True,
                        use_temp_grid=False,
                        use_greens_cache=False)

if __name__ == "__main__":
    for i in sys.argv:
        if ("=" in i):
            assert(i.count("=") == 1)
            left,right = i.split("=")
            left = left.strip()
            right = right.strip()
            if (left == "ng"):
                print("ng = " + right + ": " + str(calc(ng=float(right))[0]) + " GB")
            if (left == "size"):
                print("ng = " + str(calc(size=float(right))))