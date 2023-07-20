#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "haccgpm.hpp"

#define VerboseTransfer

int TRANSFER_CALLS = 0;
CPUTimer_t TRANSFER_TIME = 0;
CPUTimer_t TRANSFER_MPI_TIME = 0;
CPUTimer_t TRANSFER_GPU_TIME = 0;
size_t TRANSFER_BYTES = 0;

void HACCGPM::parallel::TransferParticles(HACCGPM::Params& params,HACCGPM::parallel::MemoryManager& mem, int calls){

    getIndent(calls);

    size_t bytes = 0;

    CPUTimer_t start = CPUTimer();
    
    int n_particles = params.n_particles;
    int3 local_grid_size = make_int3(params.local_grid_size[0],params.local_grid_size[1],params.local_grid_size[2]);
    int3 global_grid_size = make_int3(params.grid_dims[0],params.grid_dims[1],params.grid_dims[2]);
    int3 grid_coords = make_int3(params.grid_coords[0],params.grid_coords[1],params.grid_coords[2]);

    #ifdef VerboseTransfer
    if(params.world_rank == 0)printf("%sTransferParticles was called\n",indent);
    #endif

    int n_swaps[params.world_size];
    int n_recvs[params.world_size];
    for (int i = 0; i < params.world_size; i++){
        n_recvs[i] = 0;
    }
    int starts[params.world_size];
    
    #ifdef VerboseTransfer
    if(params.world_rank == 0)printf("%s   Loading buffers\n",indent);
    #endif

    float4* swap = (float4*)malloc(sizeof(float4)*2*n_particles);

    CPUTimer_t gpu_time = 0;
    gpu_time += HACCGPM::parallel::LoadIntoBuffers(swap,n_swaps,starts,mem.d_pos,mem.d_vel,params.nlocal,local_grid_size,grid_coords,global_grid_size,n_particles,params.ng,params.blockSize,params.world_rank,params.world_size,calls+1);
    
    bool neighbor_ranks[params.world_size];
    for (int i = 0; i < params.world_size; i++){
        neighbor_ranks[i] = false;
    }
    for (int x = -1; x < 2; x++){
        for (int y = -1; y < 2; y++){
            for (int z = -1; z < 2; z++){
                int3 neighbor_coords;
                neighbor_coords.x = ((grid_coords.x + x) + global_grid_size.x) % global_grid_size.x;
                neighbor_coords.y = ((grid_coords.y + y) + global_grid_size.y) % global_grid_size.y;
                neighbor_coords.z = ((grid_coords.z + z) + global_grid_size.z) % global_grid_size.z;

                int this_rank = neighbor_coords.x * global_grid_size.y * global_grid_size.z + neighbor_coords.y * global_grid_size.z + neighbor_coords.z;
                if (this_rank == params.world_rank)continue;
                neighbor_ranks[this_rank] = true;
            }
        }
    }
    neighbor_ranks[params.world_rank] = false;

    if(params.world_rank == 0)printf("%s   Send Counts\n",indent);
    MPI_Barrier(MPI_COMM_WORLD);
    CPUTimer_t mpi_start = CPUTimer(); 
    for (int i = 0; i < params.world_size; i++){
        if (!neighbor_ranks[i])continue;
        if (i == params.world_rank)continue;
        MPI_Request req;
        MPI_Isend(&n_swaps[i],1,MPI_INT,i,0,MPI_COMM_WORLD,&req);
        bytes += sizeof(int) * 1;
        //printf("%s      Rank %d sending n_swaps[%d] = %d to rank %d\n",indent,params.world_rank,i,n_swaps[i],i);
        MPI_Request_free(&req);
        //printf("rank %d is neighbors with %d\n",params.world_rank,i);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(params.world_rank == 0)printf("%s   Recv Counts\n",indent);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < params.world_size; i++){
        if (!neighbor_ranks[i])continue;
        if (i == params.world_rank)continue;
        MPI_Recv(&n_recvs[i],1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //printf("%s      Rank %d recieving n_recvs[%d] = %d from rank %d\n",indent,params.world_rank,i,n_recvs[i],i);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int total_recieved = 0;
    int recv_starts[params.world_size];
    for (int i = 0; i < params.world_size; i++){
        recv_starts[i] = total_recieved;
        if (i == params.world_rank)continue;
        total_recieved += n_recvs[i];
    }

    float4* recieved = (float4*)malloc(sizeof(float4)*2*n_particles);
    MPI_Barrier(MPI_COMM_WORLD);
    if(params.world_rank == 0)printf("%s   Send particles\n",indent);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < params.world_size; i++){
        if ((!neighbor_ranks[i]) || (n_swaps[i] == 0))continue;
        if (i == params.world_rank)continue;
        MPI_Request req;
        MPI_Isend(&swap[starts[i]*2],n_swaps[i]*4*2,MPI_FLOAT,i,0,MPI_COMM_WORLD,&req);
        bytes += sizeof(float) * 4 * 2 * n_swaps[i];
        //printf("%s      Rank %d sending %d particles (start = %d) to rank %d\n",indent,params.world_rank,n_swaps[i],starts[i]*2,i);
        MPI_Request_free(&req);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(params.world_rank == 0)printf("%s   Recv particles\n",indent);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < params.world_size; i++){
        if ((!neighbor_ranks[i]) || (n_recvs[i] == 0))continue;
        if (i == params.world_rank)continue;
        MPI_Recv(&recieved[recv_starts[i]*2],n_recvs[i]*4*2,MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //printf("%s      Rank %d reciving %d particles (start = %d) from rank %d\n",indent,params.world_rank,n_recvs[i],recv_starts[i]*2,i);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    CPUTimer_t mpi_end = CPUTimer();

    gpu_time += HACCGPM::parallel::insertParticles(mem.d_pos,mem.d_vel,recieved,total_recieved,params.n_particles,params.blockSize,params.world_rank,calls+1);

    free(swap);
    free(recieved);

    CPUTimer_t end = CPUTimer();
    CPUTimer_t total_time = end-start;

    TRANSFER_CALLS++;
    TRANSFER_TIME += total_time;
    TRANSFER_GPU_TIME += gpu_time;
    TRANSFER_MPI_TIME += mpi_end - mpi_start;
    TRANSFER_BYTES += bytes;


    #ifdef VerboseTransfer
    if(params.world_rank == 0)printf("%s   TransferParticles took %llu us\n",indent,total_time);
    #endif
}

void HACCGPM::parallel::sendPower(int* binCounts, double* binVals, int nbins, int world_rank, int world_size, int calls){
    getIndent(calls);
    #ifdef VerboseTransfer
    if(world_rank == 0)printf("%ssendPower was called\n",indent);
    if(world_rank == 0)printf("%s   Waiting for all\n",indent);
    #endif
    MPI_Barrier(MPI_COMM_WORLD);

    #ifdef VerboseTransfer
    if(world_rank == 0)printf("%s      Waited for all\n",indent);
    if(world_rank == 0)printf("%s   Allocating binCountsOut, binValsOut\n",indent);
    #endif

    int* binCountsOut = (int*)malloc(sizeof(int)*nbins);
    double* binValsOut = (double*)malloc(sizeof(double)*nbins);

    #ifdef VerboseTransfer
    if(world_rank == 0)printf("%s      Allocated binCountsOut, binValsOut\n",indent);
    if(world_rank == 0)printf("%s   Doing reduce\n",indent);
    #endif

    MPI_Reduce(binCounts,binCountsOut,nbins,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(binVals,binValsOut,nbins,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    #ifdef VerboseTransfer
    if(world_rank == 0)printf("%s      Done return\n",indent);
    if(world_rank == 0)printf("%s   Copying to rank 0\n",indent);
    #endif

    if(world_rank == 0){
        for (int i = 0; i < nbins; i++){
            binCounts[i] = binCountsOut[i];
            binVals[i] = binValsOut[i];
        }
    }
    
    #ifdef VerboseTransfer
    if(world_rank == 0)printf("%s      Copied to rank 0\n",indent);
    if(world_rank == 0)printf("%s   Freeing binCountsOut, binValsOut\n",indent);
    #endif

    free(binCountsOut);
    free(binValsOut);

    MPI_Barrier(MPI_COMM_WORLD);

    #ifdef VerboseTransfer
    if(world_rank == 0)printf("%s      Freed binCountsOut, binValsOut\n",indent);
    //if(world_rank == 0)printf("%s   Freeing binCountsOut, binValsOut\n",indent);
    #endif
}

void HACCGPM::parallel::printTransferBytes(int world_rank){
    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long long my_bytes = TRANSFER_BYTES;
    unsigned long long bytes;
    MPI_Reduce(&my_bytes,&bytes,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    if (world_rank != 0)return;
    printf("   transferParticles    -> calls: %10d | transferred %llu bytes\n",TRANSFER_CALLS,bytes);
}

void HACCGPM::parallel::printTransferTimes(int world_rank){
    MPI_Barrier(MPI_COMM_WORLD);
    CPUTimer_t total_min,total_max,total_mean,gpu_min,gpu_max,gpu_mean,mpi_min,mpi_max,mpi_mean,cpu_min,cpu_max,cpu_mean;
    HACCGPM::parallel::timing_stats(TRANSFER_TIME,&total_min,&total_max,&total_mean);
    HACCGPM::parallel::timing_stats(TRANSFER_GPU_TIME,&gpu_min,&gpu_max,&gpu_mean);
    HACCGPM::parallel::timing_stats(TRANSFER_MPI_TIME,&mpi_min,&mpi_max,&mpi_mean);
    HACCGPM::parallel::timing_stats(TRANSFER_TIME - (TRANSFER_GPU_TIME + TRANSFER_MPI_TIME),&cpu_min,&cpu_max,&cpu_mean);
    if (world_rank != 0)return;
    printf("   transferParticles     -> calls: %d\n",TRANSFER_CALLS);
    printf("                               total: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean,total_max,total_min);
    printf("                                 cpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",cpu_mean,cpu_max, cpu_mean);
    printf("                                 gpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",gpu_mean,gpu_max,gpu_min);
    printf("                                 mpi: %10llu us mean | %10llu us max  | %10llu us min  |\n",mpi_mean,mpi_max,mpi_min);
    if(TRANSFER_CALLS != 0)printf("                                 avg: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean / TRANSFER_CALLS,total_max / TRANSFER_CALLS,total_min / TRANSFER_CALLS);
}