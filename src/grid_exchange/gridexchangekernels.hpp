
CPUTimer_t loadXLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

CPUTimer_t storeXLeft(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

CPUTimer_t loadXRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

CPUTimer_t storeXRight(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

CPUTimer_t loadYLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

CPUTimer_t storeYLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

CPUTimer_t loadYRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

CPUTimer_t storeYRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

CPUTimer_t loadZLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

CPUTimer_t storeZLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

CPUTimer_t loadZRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

CPUTimer_t storeZRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

class XReturn{
    public:
        CPUTimer_t load(float4* h_left, float4* h_right, float4* d_in, int3 local_grid_size, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

        CPUTimer_t store(float4* h_left, float4* h_right, float4* d_in, int3 local_grid_size, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);
};

class YReturn{
    public:
        CPUTimer_t load(float4* h_left, float4* h_right, float4* d_in, int3 local_grid_size, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

        CPUTimer_t store(float4* h_left, float4* h_right, float4* d_in, int3 local_grid_size, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);
};

class ZReturn{
    public:
        CPUTimer_t load(float4* h_left, float4* h_right, float4* d_in, int3 local_grid_size, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);

        CPUTimer_t store(float4* h_left, float4* h_right, float4* d_in, int3 local_grid_size, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls);
};