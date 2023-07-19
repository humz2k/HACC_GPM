
void loadXLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

void storeXLeft(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

void loadXRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

void storeXRight(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

void loadYLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

void storeYLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

void loadYRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

void storeYRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

void loadZLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

void storeZLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

void loadZRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

void storeZRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);