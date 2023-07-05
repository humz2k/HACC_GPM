#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "bdwgc/include/gc.h"
#include "haccgpm.hpp"

#define VerboseTransfer

int TRANSFER_CALLS = 0;
CPUTimer_t TRANSFER_TIME = 0;
CPUTimer_t TRANSFER_MPI_TIME = 0;
CPUTimer_t TRANSFER_GPU_TIME = 0;
size_t TRANSFER_BYTES = 0;

int coords2rank(int3 coords, int3 dims){
    return coords.x * dims.y * dims.z + coords.y * dims.z + coords.z;
}

void zero(int* ptr, int n){
    for (int i = 0; i < n; i++){
        ptr[i] = 0;
    }
}

void sends_per_rank(int* n_sends, int* n_swaps, int3 grid_coords, int3 global_grid_size){
    int idx = 0;
    for (int x = -1; x < 2; x++){
        for (int y = -1; y < 2; y++){
            for (int z = -1; z < 2; z++){
                if (!(x == 0 && y == 0 && z == 0)){
                    int3 dest_coords = make_int3((grid_coords.x + x + global_grid_size.x)%global_grid_size.x, 
                                            (grid_coords.y + y + global_grid_size.y)%global_grid_size.y, 
                                            (grid_coords.z + z + global_grid_size.z)%global_grid_size.z);
                    int dest_rank = coords2rank(dest_coords,global_grid_size);
                    n_sends[dest_rank] += n_swaps[idx];
                }
                idx++;
            }
        }
    }
}

void combine_sends(float4** pos_sends, float4** vel_sends, float4** tmp_swap_pos, float4** tmp_swap_vel, int* n_sends, int* n_swaps, int3 grid_coords, int3 global_grid_size, int world_size, int world_rank){
    for (int i = 0; i < world_size; i++){
        if (n_sends[i] > 0){
            pos_sends[i] = (float4*)GC_MALLOC(sizeof(float4)*n_sends[i]);
            vel_sends[i] = (float4*)GC_MALLOC(sizeof(float4)*n_sends[i]);
        }
    }

    int counts[world_size];
    zero(counts,world_size);

    int idx = 0;
    for (int x = -1; x < 2; x++){
        for (int y = -1; y < 2; y++){
            for (int z = -1; z < 2; z++){
                if (!(x == 0 && y == 0 && z == 0)){
                    int3 dest_coords = make_int3((grid_coords.x + x + global_grid_size.x)%global_grid_size.x, 
                                            (grid_coords.y + y + global_grid_size.y)%global_grid_size.y, 
                                            (grid_coords.z + z + global_grid_size.z)%global_grid_size.z);
                    int dest_rank = coords2rank(dest_coords,global_grid_size);
                    for (int i = 0; i < n_swaps[idx]; i++){
                        float4 this_particle = tmp_swap_pos[idx][i];
                        float4 this_vel = tmp_swap_vel[idx][i];
                        int store_idx = counts[dest_rank];
                        counts[dest_rank]++;
                        pos_sends[dest_rank][store_idx] = this_particle;
                        vel_sends[dest_rank][store_idx] = this_vel;
                    }
                }
                idx++;
            }
        }
    }
}

CPUTimer_t send_recv_data(int world_size, int world_rank, int* n_sends, int* n_recvs, float4** pos_recvs, float4** vel_recvs, float4** pos_sends, float4** vel_sends, int calls){
    getIndent(calls);
    #ifdef VerboseTransfer
    if(world_rank == 0)printf("%ssend_recv_data was called\n",indent);
    if(world_rank == 0)printf("%s   Sending/Recieving counts\n",indent);
    #endif
    CPUTimer_t start_counts = CPUTimer();
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if (i != world_rank)MPI_Send(&n_sends[i],1,MPI_INT,i,0,MPI_COMM_WORLD);
        TRANSFER_BYTES += n_sends[i] * sizeof(int);
    }
    for (int i = 0; i < world_size; i++){
        if (i != world_rank)MPI_Recv(&n_recvs[i],1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    CPUTimer_t end_counts = CPUTimer();
    #ifdef VerboseTransfer
    /*if(world_rank == 0)printf("%s      Sent/Recieved counts\n",indent);
    for (int i = 0; i < world_size; i++){
        if (world_rank == i){
            //printf("%s   Rank %d\n",indent,world_rank);
            for (int j = 0; j < world_size; j++){
                //if (j == world_rank)continue;
                printf("%s      Rank %d send %5d to   Rank %4d\n",indent,world_rank,n_sends[j],j);
            }
            for (int j = 0; j < world_size; j++){
                //if (j == world_rank)continue;
                printf("%s      Rank %d recv %5d from Rank %4d\n",indent,world_rank,n_recvs[j],j);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }*/
    #endif
    CPUTimer_t start_particles = CPUTimer();
    for (int i = 0; i < world_size; i++){
        if (n_recvs[i] > 0){
            pos_recvs[i] = (float4*)GC_MALLOC(sizeof(float4)*n_recvs[i]);
            vel_recvs[i] = (float4*)GC_MALLOC(sizeof(float4)*n_recvs[i]);
        }
    }
    #ifdef VerboseTransfer
    if(world_rank == 0)printf("%s   Sending/Recieving particles\n",indent);
    #endif
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if (i != world_rank){
            if (n_sends[i] != 0){
                MPI_Send(pos_sends[i],n_sends[i]*4,MPI_FLOAT,i,0,MPI_COMM_WORLD);
                MPI_Send(vel_sends[i],n_sends[i]*4,MPI_FLOAT,i,1,MPI_COMM_WORLD);
                TRANSFER_BYTES += n_sends[i] * 4 * sizeof(float) * 2;
            }
        }
    }
    for (int i = 0; i < world_size; i++){
        if (i != world_rank){
            if (n_recvs[i] != 0){
                MPI_Recv(pos_recvs[i],n_recvs[i]*4,MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Recv(vel_recvs[i],n_recvs[i]*4,MPI_FLOAT,i,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef VerboseTransfer
    if(world_rank == 0)printf("%s      Sent/Recieved particles\n",indent);
    #endif
    CPUTimer_t end_particles = CPUTimer();
    CPUTimer_t count_time = end_counts - start_counts;
    CPUTimer_t particle_time = end_particles - start_particles;
    CPUTimer_t total_time = particle_time + count_time;
    if(world_rank == 0)printf("%s   send_recv_data took %llu us\n",indent,total_time);
    return total_time;
}

void HACCGPM::parallel::transferParticles(HACCGPM::Params& params,HACCGPM::parallel::MemoryManager& mem, int calls){

    getIndent(calls);

    CPUTimer_t start = CPUTimer();

    int n_particles = params.frac * params.nlocal;
    int3 local_grid_size = make_int3(params.local_grid_size[0],params.local_grid_size[1],params.local_grid_size[2]);
    int3 global_grid_size = make_int3(params.grid_dims[0],params.grid_dims[1],params.grid_dims[2]);
    int3 grid_coords = make_int3(params.grid_coords[0],params.grid_coords[1],params.grid_coords[2]);

    #ifdef VerboseTransfer
    if(params.world_rank == 0)printf("%stransferParticles was called\n",indent);
    #endif

    float4* tmp_swap_pos[27];
    float4* tmp_swap_vel[27];

    int n_swaps[27];
    
    #ifdef VerboseTransfer
    if(params.world_rank == 0)printf("%s   Loading buffers\n",indent);
    #endif

    CPUTimer_t gpu_time = 0;
    gpu_time += HACCGPM::parallel::loadIntoBuffers(tmp_swap_pos,tmp_swap_vel,n_swaps,mem.d_pos,mem.d_vel,params.nlocal,local_grid_size,n_particles,params.blockSize,params.world_rank,calls+1);

    #ifdef VerboseTransfer
    if(params.world_rank == 0)printf("%s      Loaded buffers\n",indent);
    #endif

    int remaining = n_swaps[13];

    int n_recvs[params.world_size];
    zero(n_recvs,params.world_size);
    int n_sends[params.world_size];
    zero(n_sends,params.world_size);
    float4* pos_recvs[params.world_size];
    float4* vel_recvs[params.world_size];
    float4* pos_sends[params.world_size];
    float4* vel_sends[params.world_size];

    sends_per_rank(n_sends,n_swaps,grid_coords,global_grid_size);

    combine_sends(pos_sends,vel_sends,tmp_swap_pos,tmp_swap_vel,n_sends,n_swaps,grid_coords,global_grid_size,params.world_size,params.world_rank);
    
    CPUTimer_t mpi_time = send_recv_data(params.world_size,params.world_rank,n_sends,n_recvs,pos_recvs,vel_recvs,pos_sends,vel_sends,calls+1);

    int total_recieved = 0;
    for (int i = 0; i < params.world_size; i++){
        total_recieved += n_recvs[i];
    }

    #ifdef VerboseTransfer
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < params.world_size; i++){
        
        if(params.world_rank == i)printf("%s   Rank %d: Total Recieved = %d\n",indent,params.world_rank,total_recieved);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    float4* pos_to_transfer = (float4*)malloc(sizeof(float4)*total_recieved);
    float4* vel_to_transfer = (float4*)malloc(sizeof(float4)*total_recieved);

    int idx = 0;
    for (int i = 0; i < params.world_size; i++){
        if (i != params.world_rank){
            if (n_recvs[i] != 0){
                for (int j = 0; j < n_recvs[i]; j++){
                    if (idx >= total_recieved){
                        printf("????\n");
                    }
                    pos_to_transfer[idx] = pos_recvs[i][j];
                    vel_to_transfer[idx] = vel_recvs[i][j];
                    idx++;
                }
            }
        }
    }

    gpu_time += HACCGPM::parallel::insertParticles(mem.d_pos,mem.d_vel,pos_to_transfer,vel_to_transfer,total_recieved,remaining,params.n_particles,params.blockSize,params.world_rank,calls+1);

    free(pos_to_transfer);
    free(vel_to_transfer);

    CPUTimer_t end = CPUTimer();
    CPUTimer_t total_time = end-start;

    TRANSFER_CALLS++;
    TRANSFER_TIME += total_time;
    TRANSFER_GPU_TIME += gpu_time;
    TRANSFER_MPI_TIME += mpi_time;


    #ifdef VerboseTransfer
    if(params.world_rank == 0)printf("%s   transferParticles took %llu us\n",indent,total_time);
    #endif
}

void HACCGPM::parallel::printTransferBytes(int world_rank){
    MPI_Barrier(MPI_COMM_WORLD);
    unsigned long long my_bytes = TRANSFER_BYTES;
    unsigned long long bytes;
    MPI_Reduce(&my_bytes,&bytes,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    if (world_rank != 0)return;
    printf("   transferParticles -> calls: %10d | transferred %llu bytes\n",TRANSFER_CALLS,bytes);
}

void HACCGPM::parallel::printTransferTimes(int world_rank){
    MPI_Barrier(MPI_COMM_WORLD);
    CPUTimer_t total_min,total_max,total_mean,gpu_min,gpu_max,gpu_mean,mpi_min,mpi_max,mpi_mean;
    HACCGPM::parallel::timing_stats(TRANSFER_TIME,&total_min,&total_max,&total_mean);
    HACCGPM::parallel::timing_stats(TRANSFER_GPU_TIME,&gpu_min,&gpu_max,&gpu_mean);
    HACCGPM::parallel::timing_stats(TRANSFER_MPI_TIME,&mpi_min,&mpi_max,&mpi_mean);
    if (world_rank != 0)return;
    printf("   transferParticles  -> calls: %10d\n",TRANSFER_CALLS);
    printf("                               total: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean,total_max,total_min);
    printf("                                 cpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",(total_mean-gpu_mean) - mpi_mean,(total_max - gpu_max) - mpi_max, (total_min - gpu_min) - mpi_min);
    printf("                                 gpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",gpu_mean,gpu_max,gpu_min);
    printf("                                 mpi: %10llu us mean | %10llu us max  | %10llu us min  |\n",mpi_mean,mpi_max,mpi_min);
    printf("                                 avg: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean / TRANSFER_CALLS,total_max / TRANSFER_CALLS,total_min / TRANSFER_CALLS);
}