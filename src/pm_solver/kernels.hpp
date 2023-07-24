
#include "haccgpm.hpp"

void getGreens(hostFFT_t* __restrict d_greens, int ng, int numBlocks, int blockSize, int calls);

void getGreens(hostFFT_t* __restrict d_greens, int ng, int nlocal, int3 local_grid_size_vec, int3 grid_coords_vec, int world_rank, int numBlocks, int blockSize,  int calls);