#include <stdlib.h>
#include <stdio.h>
//#include <cstdlib>
#include <mpi.h>
#include "haccgpm.hpp"
#include "gridexchangekernels.hpp"

int RESOLVE_CALLS = 0;
CPUTimer_t RESOLVE_TIME = 0;
CPUTimer_t RESOLVE_MPI_TIME = 0;
CPUTimer_t RESOLVE_GPU_TIME = 0;
size_t RESOLVE_BYTES = 0;

HACCGPM::parallel::GridExchange::GridExchange(){};

HACCGPM::parallel::GridExchange::GridExchange(int3 local_coords_, int3 local_grid_size_, int3 dims_, int ng_, int world_size_, int world_rank_, int overload_, int blockSize_) : local_coords(local_coords_),
                                                                                                                    local_grid_size(local_grid_size_),
                                                                                                                    ng(ng_),
                                                                                                                    world_size(world_size_),
                                                                                                                    world_rank(world_rank_),
                                                                                                                    overload(overload_),
                                                                                                                    blockSize(blockSize_),
                                                                                                                    dims(dims_){};


//#define VerboseResolveSends

//#define VerboseResolve

template<class ThisDim>
void do_return(float4* grad, int left_rank, int right_rank, int size, int3 local_grid_size, int3 local_coords, int3 total_grid_dims, int overload, int world_rank, size_t* bytes, CPUTimer_t* mpi_time, int blockSize, int calls){

    getIndent(calls);

    //size = total_grid_dims.x * overload * local_grid_size.z;
    float4* left_x_sends = (float4*)malloc(sizeof(float4)*size);
    float4* left_x_recvs = (float4*)malloc(sizeof(float4)*size);

    float4* right_x_sends = (float4*)malloc(sizeof(float4)*size);
    float4* right_x_recvs = (float4*)malloc(sizeof(float4)*size);

    ThisDim this_dim;

    this_dim.load(left_x_sends,right_x_sends,grad,local_grid_size,total_grid_dims,overload,size,blockSize,world_rank,calls);

    if(world_rank == 0)printf("%sSending...\n",indent);

    CPUTimer_t mpi_start = CPUTimer();

    {
    MPI_Request req;
    MPI_Isend(left_x_sends,size*4,MPI_FLOAT,left_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    {
    MPI_Request req;
    MPI_Isend(right_x_sends,size*4,MPI_FLOAT,right_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    *bytes += 2*size*sizeof(float4);

    if(world_rank == 0)printf("%s   Sent %lu bytes\n",indent,2*size*sizeof(float4));
    if(world_rank == 0)printf("%sRecieving...\n",indent);

    MPI_Recv(left_x_recvs,size*4,MPI_FLOAT,right_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    MPI_Recv(right_x_recvs,size*4,MPI_FLOAT,left_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    if(world_rank == 0)printf("%s   Recieved %lu bytes\n",indent,2*size*sizeof(float4));

    CPUTimer_t mpi_end = CPUTimer();
    *mpi_time += mpi_end - mpi_start;

    MPI_Barrier(MPI_COMM_WORLD);

    free(left_x_sends);
    free(right_x_sends);

    this_dim.store(left_x_recvs,right_x_recvs,grad,local_grid_size,total_grid_dims,overload,size,blockSize,world_rank,calls);

    free(left_x_recvs);
    free(right_x_recvs);

    MPI_Barrier(MPI_COMM_WORLD);
}

void HACCGPM::parallel::GridExchange::fill(float4* grad, int calls){
    getIndent(calls);

    CPUTimer_t mpi_start,mpi_end;
    CPUTimer_t mpi_time = 0;
    size_t bytes = 0;

    #ifdef VerboseResolve
    if(world_rank == 0)printf("%sDoing Grid Exchange Fill (called GridExchange.fill)\n",indent);
    #endif

    int3 total_grid_dims = make_int3(local_grid_size.x + 2*overload,local_grid_size.y + 2*overload,local_grid_size.z + 2*overload);
    int total_grid_size = total_grid_dims.x*total_grid_dims.y*total_grid_dims.z;

    #ifdef VerboseResolve
    if(world_rank == 0)printf("%s   Overload Dimensions: [%d %d %d]\n",indent,total_grid_dims.x,total_grid_dims.y,total_grid_dims.z);
    #endif
    int size, left_rank, right_rank;
    int3 left_coords, right_coords;

    size = overload * local_grid_size.y * local_grid_size.z;

    left_coords = make_int3((local_coords.x-1 + dims.x)%dims.x,local_coords.y,local_coords.z);
    left_rank = left_coords.x * dims.y * dims.z + left_coords.y * dims.z + left_coords.z;

    right_coords = make_int3((local_coords.x+1 + dims.x)%dims.x,local_coords.y,local_coords.z);
    right_rank = right_coords.x * dims.y * dims.z + right_coords.y * dims.z + right_coords.z;
    do_return<XReturn>(grad,left_rank,right_rank,size,local_grid_size,local_coords,total_grid_dims,overload,world_rank,&bytes,&mpi_time,blockSize,calls+1);


    size = total_grid_dims.x * overload * local_grid_size.z;

    left_coords = make_int3(local_coords.x,(local_coords.y-1 + dims.y)%dims.y,local_coords.z);
    left_rank = left_coords.x * dims.y * dims.z + left_coords.y * dims.z + left_coords.z;

    right_coords = make_int3(local_coords.x,(local_coords.y+1 + dims.y)%dims.y,local_coords.z);
    right_rank = right_coords.x * dims.y * dims.z + right_coords.y * dims.z + right_coords.z;
    do_return<YReturn>(grad,left_rank,right_rank,size,local_grid_size,local_coords,total_grid_dims,overload,world_rank,&bytes,&mpi_time,blockSize,calls+1);


    size = total_grid_dims.x * total_grid_dims.y * overload;

    left_coords = make_int3(local_coords.x,local_coords.y,(local_coords.z-1 + dims.z)%dims.z);
    left_rank = left_coords.x * dims.y * dims.z + left_coords.y * dims.z + left_coords.z;

    right_coords = make_int3(local_coords.x,local_coords.y,(local_coords.z+1 + dims.z)%dims.z);
    right_rank = right_coords.x * dims.y * dims.z + right_coords.y * dims.z + right_coords.z;
    do_return<ZReturn>(grad,left_rank,right_rank,size,local_grid_size,local_coords,total_grid_dims,overload,world_rank,&bytes,&mpi_time,blockSize,calls+1);

}

void HACCGPM::parallel::GridExchange::resolve(float* grid, int calls){
    getIndent(calls);

    #ifdef VerboseResolve
    if(world_rank == 0)printf("%sDoing Grid Exchange Resolve (called GridExchange.resolve)\n",indent);
    #endif

    CPUTimer_t mpi_start,mpi_end;

    CPUTimer_t mpi_time = 0;
    CPUTimer_t gpu_time = 0;

    size_t bytes = 0;

    CPUTimer_t start = CPUTimer();

    int3 total_grid_dims = make_int3(local_grid_size.x + 2*overload,local_grid_size.y + 2*overload,local_grid_size.z + 2*overload);
    int total_grid_size = total_grid_dims.x*total_grid_dims.y*total_grid_dims.z;
    #ifdef VerboseResolve
    if(world_rank == 0)printf("%s   Overload Dimensions: [%d %d %d]\n",indent,total_grid_dims.x,total_grid_dims.y,total_grid_dims.z);
    #endif
    int size, dest_rank, recv_rank;
    int3 dest_coords, recv_coords;

    //X DIMENSION

    size = total_grid_dims.y * total_grid_dims.z * overload;
    
    dest_coords = make_int3((local_coords.x-1 + dims.x)%dims.x,local_coords.y,local_coords.z);
    dest_rank = dest_coords.x * dims.y * dims.z + dest_coords.y * dims.z + dest_coords.z;

    #ifdef VerboseResolveSends
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if(i == world_rank)printf("%s      Rank %d ([%d %d %d]): Send X (left) with size = %d and dest = %d ([%d %d %d])\n",indent,world_rank,local_coords.x,local_coords.y,local_coords.z,size,dest_rank,dest_coords.x,dest_coords.y,dest_coords.z);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    recv_coords = make_int3((local_coords.x+1)%dims.x,local_coords.y,local_coords.z);
    recv_rank = recv_coords.x * dims.y * dims.z + recv_coords.y * dims.z + recv_coords.z;

    #ifdef VerboseResolveSends
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if(i == world_rank)printf("%s      Rank %d ([%d %d %d]): Recv X (left) with size = %d and recv = %d ([%d %d %d])\n",indent,world_rank,local_coords.x,local_coords.y,local_coords.z,size,recv_rank,recv_coords.x,recv_coords.y,recv_coords.z);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
    
    float* left_x_sends = (float*)malloc(sizeof(float)*size);
    float* left_x_recvs = (float*)malloc(sizeof(float)*size);
    gpu_time += loadXLeft(left_x_sends,grid,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    mpi_start = CPUTimer();

    {
    MPI_Request req;
    MPI_Isend(left_x_sends,size,MPI_FLOAT,dest_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    bytes += size*sizeof(float);

    MPI_Recv(left_x_recvs,size,MPI_FLOAT,recv_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    mpi_end = CPUTimer();
    mpi_time += mpi_end - mpi_start;

    gpu_time += storeXLeft(grid,left_x_recvs,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    mpi_start = CPUTimer();

    MPI_Barrier(MPI_COMM_WORLD);

    mpi_end = CPUTimer();
    mpi_time += mpi_end - mpi_start;

    free(left_x_sends);
    free(left_x_recvs);


    dest_coords = make_int3((local_coords.x+1 + dims.x)%dims.x,local_coords.y,local_coords.z);
    dest_rank = dest_coords.x * dims.y * dims.z + dest_coords.y * dims.z + dest_coords.z;

    #ifdef VerboseResolveSends
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if(i == world_rank)printf("%s      Rank %d ([%d %d %d]): Send X (left) with size = %d and dest = %d ([%d %d %d])\n",indent,world_rank,local_coords.x,local_coords.y,local_coords.z,size,dest_rank,dest_coords.x,dest_coords.y,dest_coords.z);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    recv_coords = make_int3((local_coords.x-1 + dims.x)%dims.x,local_coords.y,local_coords.z);
    recv_rank = recv_coords.x * dims.y * dims.z + recv_coords.y * dims.z + recv_coords.z;

    #ifdef VerboseResolveSends
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if(i == world_rank)printf("%s      Rank %d ([%d %d %d]): Recv X (left) with size = %d and recv = %d ([%d %d %d])\n",indent,world_rank,local_coords.x,local_coords.y,local_coords.z,size,recv_rank,recv_coords.x,recv_coords.y,recv_coords.z);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    float* right_x_sends = (float*)malloc(sizeof(float)*size);
    float* right_x_recvs = (float*)malloc(sizeof(float)*size);
    gpu_time += loadXRight(right_x_sends,grid,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    mpi_start = CPUTimer();

    {
    MPI_Request req;
    MPI_Isend(right_x_sends,size,MPI_FLOAT,dest_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    bytes += size*sizeof(float);

    MPI_Recv(right_x_recvs,size,MPI_FLOAT,recv_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    mpi_end = CPUTimer();
    mpi_time += mpi_end - mpi_start;

    gpu_time += storeXRight(grid,right_x_recvs,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    mpi_start = CPUTimer();

    MPI_Barrier(MPI_COMM_WORLD);
    
    mpi_end = CPUTimer();
    mpi_time += mpi_end - mpi_start;

    free(right_x_sends);
    free(right_x_recvs);

    //Y Left

    size = (total_grid_dims.x - 2*overload) * total_grid_dims.z * overload;

    dest_coords = make_int3(local_coords.x,(local_coords.y-1 + dims.y)%dims.y,local_coords.z);
    dest_rank = dest_coords.x * dims.y * dims.z + dest_coords.y * dims.z + dest_coords.z;

    #ifdef VerboseResolveSends
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if(i == world_rank)printf("%s      Rank %d ([%d %d %d]): Send X (left) with size = %d and dest = %d ([%d %d %d])\n",indent,world_rank,local_coords.x,local_coords.y,local_coords.z,size,dest_rank,dest_coords.x,dest_coords.y,dest_coords.z);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    recv_coords = make_int3(local_coords.x,(local_coords.y+1 + dims.y)%dims.y,local_coords.z);
    recv_rank = recv_coords.x * dims.y * dims.z + recv_coords.y * dims.z + recv_coords.z;

    #ifdef VerboseResolveSends
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if(i == world_rank)printf("%s      Rank %d ([%d %d %d]): Recv X (left) with size = %d and recv = %d ([%d %d %d])\n",indent,world_rank,local_coords.x,local_coords.y,local_coords.z,size,recv_rank,recv_coords.x,recv_coords.y,recv_coords.z);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    float* left_y_sends = (float*)malloc(sizeof(float)*size);
    float* left_y_recvs = (float*)malloc(sizeof(float)*size);
    gpu_time += loadYLeft(left_y_sends,grid,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    mpi_start = CPUTimer();

    {
    MPI_Request req;
    MPI_Isend(left_y_sends,size,MPI_FLOAT,dest_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    bytes += size*sizeof(float);

    MPI_Recv(left_y_recvs,size,MPI_FLOAT,recv_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    mpi_end = CPUTimer();
    mpi_time += mpi_end - mpi_start;

    gpu_time += storeYLeft(grid,left_y_recvs,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    mpi_start = CPUTimer();

    MPI_Barrier(MPI_COMM_WORLD);

    mpi_end = CPUTimer();
    mpi_time += mpi_end - mpi_start;

    free(left_y_sends);
    free(left_y_recvs);


    //Y right

    //size = (total_grid_dims.x - 2*overload) * total_grid_dims.z * overload;

    dest_coords = make_int3(local_coords.x,(local_coords.y+1 + dims.y)%dims.y,local_coords.z);
    dest_rank = dest_coords.x * dims.y * dims.z + dest_coords.y * dims.z + dest_coords.z;

    #ifdef VerboseResolveSends
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if(i == world_rank)printf("%s      Rank %d ([%d %d %d]): Send X (left) with size = %d and dest = %d ([%d %d %d])\n",indent,world_rank,local_coords.x,local_coords.y,local_coords.z,size,dest_rank,dest_coords.x,dest_coords.y,dest_coords.z);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    recv_coords = make_int3(local_coords.x,(local_coords.y-1 + dims.y)%dims.y,local_coords.z);
    recv_rank = recv_coords.x * dims.y * dims.z + recv_coords.y * dims.z + recv_coords.z;

    #ifdef VerboseResolveSends
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if(i == world_rank)printf("%s      Rank %d ([%d %d %d]): Recv X (left) with size = %d and recv = %d ([%d %d %d])\n",indent,world_rank,local_coords.x,local_coords.y,local_coords.z,size,recv_rank,recv_coords.x,recv_coords.y,recv_coords.z);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    float* right_y_sends = (float*)malloc(sizeof(float)*size);
    float* right_y_recvs = (float*)malloc(sizeof(float)*size);
    gpu_time += loadYRight(right_y_sends,grid,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    mpi_start = CPUTimer();

    {
    MPI_Request req;
    MPI_Isend(right_y_sends,size,MPI_FLOAT,dest_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    bytes += size*sizeof(float);

    MPI_Recv(right_y_recvs,size,MPI_FLOAT,recv_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    mpi_end = CPUTimer();
    mpi_time += mpi_end - mpi_start;

    gpu_time += storeYRight(grid,right_y_recvs,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    mpi_start = CPUTimer();

    MPI_Barrier(MPI_COMM_WORLD);

    mpi_end = CPUTimer();
    mpi_time += mpi_end - mpi_start;

    free(right_y_sends);
    free(right_y_recvs);




    //Z Left

    size = (total_grid_dims.x - 2*overload) * (total_grid_dims.y - 2*overload) * overload;

    dest_coords = make_int3(local_coords.x,local_coords.y,(local_coords.z-1 + dims.z)%dims.z);
    dest_rank = dest_coords.x * dims.y * dims.z + dest_coords.y * dims.z + dest_coords.z;

    #ifdef VerboseResolveSends
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if(i == world_rank)printf("%s      Rank %d ([%d %d %d]): Send X (left) with size = %d and dest = %d ([%d %d %d])\n",indent,world_rank,local_coords.x,local_coords.y,local_coords.z,size,dest_rank,dest_coords.x,dest_coords.y,dest_coords.z);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    recv_coords = make_int3(local_coords.x,local_coords.y,(local_coords.z+1 + dims.z)%dims.z);
    recv_rank = recv_coords.x * dims.y * dims.z + recv_coords.y * dims.z + recv_coords.z;

    #ifdef VerboseResolveSends
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if(i == world_rank)printf("%s      Rank %d ([%d %d %d]): Recv X (left) with size = %d and recv = %d ([%d %d %d])\n",indent,world_rank,local_coords.x,local_coords.y,local_coords.z,size,recv_rank,recv_coords.x,recv_coords.y,recv_coords.z);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    float* left_z_sends = (float*)malloc(sizeof(float)*size);
    float* left_z_recvs = (float*)malloc(sizeof(float)*size);
    gpu_time += loadZLeft(left_z_sends,grid,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    mpi_start = CPUTimer();

    {
    MPI_Request req;
    MPI_Isend(left_z_sends,size,MPI_FLOAT,dest_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    bytes += size*sizeof(float);

    MPI_Recv(left_z_recvs,size,MPI_FLOAT,recv_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    mpi_end = CPUTimer();
    mpi_time += mpi_end - mpi_start;

    gpu_time += storeZLeft(grid,left_z_recvs,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    mpi_start = CPUTimer();

    MPI_Barrier(MPI_COMM_WORLD);

    mpi_end = CPUTimer();
    mpi_time += mpi_end - mpi_start;

    free(left_z_sends);
    free(left_z_recvs);


    //Y right

    //size = (total_grid_dims.x - 2*overload) * total_grid_dims.z * overload;

    dest_coords =  make_int3(local_coords.x,local_coords.y,(local_coords.z+1 + dims.z)%dims.z);
    dest_rank = dest_coords.x * dims.y * dims.z + dest_coords.y * dims.z + dest_coords.z;

    #ifdef VerboseResolveSends
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if(i == world_rank)printf("%s      Rank %d ([%d %d %d]): Send X (left) with size = %d and dest = %d ([%d %d %d])\n",indent,world_rank,local_coords.x,local_coords.y,local_coords.z,size,dest_rank,dest_coords.x,dest_coords.y,dest_coords.z);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    recv_coords =  make_int3(local_coords.x,local_coords.y,(local_coords.z-1 + dims.z)%dims.z);
    recv_rank = recv_coords.x * dims.y * dims.z + recv_coords.y * dims.z + recv_coords.z;

    #ifdef VerboseResolveSends
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if(i == world_rank)printf("%s      Rank %d ([%d %d %d]): Recv X (left) with size = %d and recv = %d ([%d %d %d])\n",indent,world_rank,local_coords.x,local_coords.y,local_coords.z,size,recv_rank,recv_coords.x,recv_coords.y,recv_coords.z);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    float* right_z_sends = (float*)malloc(sizeof(float)*size);
    float* right_z_recvs = (float*)malloc(sizeof(float)*size);
    gpu_time += loadZRight(right_z_sends,grid,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    mpi_start = CPUTimer();

    {
    MPI_Request req;
    MPI_Isend(right_z_sends,size,MPI_FLOAT,dest_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    bytes += size*sizeof(float);

    MPI_Recv(right_z_recvs,size,MPI_FLOAT,recv_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    mpi_end = CPUTimer();
    mpi_time += mpi_end - mpi_start;

    gpu_time += storeZRight(grid,right_z_recvs,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    mpi_start = CPUTimer();

    MPI_Barrier(MPI_COMM_WORLD);

    mpi_end = CPUTimer();
    mpi_time += mpi_end - mpi_start;

    free(right_z_sends);
    free(right_z_recvs);

    CPUTimer_t end = CPUTimer();

    RESOLVE_CALLS++;
    RESOLVE_TIME += end-start;
    RESOLVE_MPI_TIME += mpi_time;
    RESOLVE_GPU_TIME += gpu_time;
    RESOLVE_BYTES += bytes;

    #ifdef VerboseResolve
    if(world_rank == 0)printf("%s   Grid Exchange Resolve took %llu us\n",indent,end-start);
    #endif

}

void HACCGPM::parallel::printGridExchangeBytes(int world_rank){
    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long long my_bytes = RESOLVE_BYTES;
    unsigned long long bytes;
    MPI_Reduce(&my_bytes,&bytes,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    if (world_rank != 0)return;
    printf("   gridExchange.resolve -> calls: %10d | transferred %llu bytes\n",RESOLVE_CALLS,bytes);
}

void HACCGPM::parallel::printGridExchangeTimes(int world_rank){
    MPI_Barrier(MPI_COMM_WORLD);
    CPUTimer_t total_min,total_max,total_mean,gpu_min,gpu_max,gpu_mean,mpi_min,mpi_max,mpi_mean,cpu_min,cpu_max,cpu_mean;
    HACCGPM::parallel::timing_stats(RESOLVE_TIME,&total_min,&total_max,&total_mean);
    HACCGPM::parallel::timing_stats(RESOLVE_GPU_TIME,&gpu_min,&gpu_max,&gpu_mean);
    HACCGPM::parallel::timing_stats(RESOLVE_MPI_TIME,&mpi_min,&mpi_max,&mpi_mean);
    HACCGPM::parallel::timing_stats(RESOLVE_TIME - (RESOLVE_GPU_TIME + RESOLVE_MPI_TIME),&cpu_min,&cpu_max,&cpu_mean);
    if (world_rank != 0)return;
    printf("   gridExchange.resolve  -> calls: %d\n",RESOLVE_CALLS);
    printf("                               total: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean,total_max,total_min);
    printf("                                 cpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",cpu_mean,cpu_max, cpu_min);
    printf("                                 gpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",gpu_mean,gpu_max,gpu_min);
    printf("                                 mpi: %10llu us mean | %10llu us max  | %10llu us min  |\n",mpi_mean,mpi_max,mpi_min);
    printf("                                 avg: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean / RESOLVE_CALLS,total_max / RESOLVE_CALLS,total_min / RESOLVE_CALLS);
}

