#include <stdlib.h>
#include <stdio.h>
//#include <cstdlib>
#include <mpi.h>
#include "../src/haccgpm.hpp"
#include "gridexchangekernels.hpp"

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

void HACCGPM::parallel::GridExchange::resolve(float* grid, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sDoing Grid Exchange (called GridExchange.resolve)\n",indent);

    int3 total_grid_dims = make_int3(local_grid_size.x + 2*overload,local_grid_size.y + 2*overload,local_grid_size.z + 2*overload);
    int total_grid_size = total_grid_dims.x*total_grid_dims.y*total_grid_dims.z;
    if(world_rank == 0)printf("%s   Overload Dimensions: [%d %d %d]\n",indent,total_grid_dims.x,total_grid_dims.y,total_grid_dims.z);
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
    loadXLeft(left_x_sends,grid,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    {
    MPI_Request req;
    MPI_Isend(left_x_sends,size,MPI_FLOAT,dest_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    MPI_Recv(left_x_recvs,size,MPI_FLOAT,recv_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    storeXLeft(grid,left_x_recvs,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    MPI_Barrier(MPI_COMM_WORLD);

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
    loadXRight(right_x_sends,grid,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    {
    MPI_Request req;
    MPI_Isend(right_x_sends,size,MPI_FLOAT,dest_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    MPI_Recv(right_x_recvs,size,MPI_FLOAT,recv_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    storeXRight(grid,right_x_recvs,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    MPI_Barrier(MPI_COMM_WORLD);

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
    loadYLeft(left_y_sends,grid,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    {
    MPI_Request req;
    MPI_Isend(left_y_sends,size,MPI_FLOAT,dest_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    MPI_Recv(left_y_recvs,size,MPI_FLOAT,recv_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    storeYLeft(grid,left_y_recvs,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    MPI_Barrier(MPI_COMM_WORLD);

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
    loadYRight(right_y_sends,grid,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    {
    MPI_Request req;
    MPI_Isend(right_y_sends,size,MPI_FLOAT,dest_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    MPI_Recv(right_y_recvs,size,MPI_FLOAT,recv_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    storeYRight(grid,right_y_recvs,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    MPI_Barrier(MPI_COMM_WORLD);

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
    loadZLeft(left_z_sends,grid,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    {
    MPI_Request req;
    MPI_Isend(left_z_sends,size,MPI_FLOAT,dest_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    MPI_Recv(left_z_recvs,size,MPI_FLOAT,recv_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    storeZLeft(grid,left_z_recvs,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    MPI_Barrier(MPI_COMM_WORLD);

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
    loadZRight(right_z_sends,grid,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    {
    MPI_Request req;
    MPI_Isend(right_z_sends,size,MPI_FLOAT,dest_rank,0,MPI_COMM_WORLD,&req);
    MPI_Request_free(&req);
    }

    MPI_Recv(right_z_recvs,size,MPI_FLOAT,recv_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    storeZRight(grid,right_z_recvs,total_grid_dims,overload,size,blockSize,world_rank,calls+1);

    MPI_Barrier(MPI_COMM_WORLD);

    free(right_z_sends);
    free(right_z_recvs);

}

