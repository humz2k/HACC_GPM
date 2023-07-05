#include <stdio.h>
#include <stdlib.h>
#include "haccgpm.hpp"


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

void send_recv_data(int world_size, int world_rank, int* n_sends, int* n_recvs, float4** pos_recvs, float4** vel_recvs, float4** pos_sends, float4** vel_sends){
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if (i != world_rank)MPI_Send(&n_sends[i],1,MPI_INT,i,0,MPI_COMM_WORLD);
    }
    for (int i = 0; i < world_size; i++){
        if (i != world_rank)MPI_Recv(&n_recvs[i],1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if (n_recvs[i] > 0){
            pos_recvs[i] = (float4*)GC_MALLOC(sizeof(float4)*n_recvs[i]);
            vel_recvs[i] = (float4*)GC_MALLOC(sizeof(float4)*n_recvs[i]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if (i != world_rank){
            if (n_sends[i] != 0){
                MPI_Send(pos_sends[i],n_sends[i]*4,MPI_FLOAT,i,0,MPI_COMM_WORLD);
                MPI_Send(vel_sends[i],n_sends[i]*4,MPI_FLOAT,i,1,MPI_COMM_WORLD);
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
}

void transfer_particles(HACCGPM::Params& params,HACCGPM::parallel::MemoryManager& mem){

    int n_particles = params.frac * params.nlocal;
    int3 local_grid_size = make_int3(params.local_grid_size[0],params.local_grid_size[1],params.local_grid_size[2]);
    int3 global_grid_size = make_int3(params.grid_dims[0],params.grid_dims[1],params.grid_dims[2]);
    int3 grid_coords = make_int3(params.grid_coords[0],params.grid_coords[1],params.grid_coords[2]);

    //MPI_Barrier(MPI_COMM_WORLD);

    if(params.world_rank == 0)printf("DOING SWAP!!!\n");

    float4* tmp_swap_pos[27];
    float4* tmp_swap_vel[27];

    int n_swaps[27];

    HACCGPM::parallel::loadIntoBuffers(tmp_swap_pos,tmp_swap_vel,n_swaps,mem.d_pos,mem.d_vel,params.nlocal,local_grid_size,n_particles,params.blockSize,params.world_rank);

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
    
    send_recv_data(params.world_size,params.world_rank,n_sends,n_recvs,pos_recvs,vel_recvs,pos_sends,vel_sends);

    int total_recieved = 0;
    for (int i = 0; i < params.world_size; i++){
        total_recieved += n_recvs[i];
    }

    printf("Rank %d: Total Recieved = %d\n",params.world_rank,total_recieved);

    float4* pos_to_transfer = (float4*)malloc(sizeof(float4)*total_recieved);
    float4* vel_to_transfer = (float4*)malloc(sizeof(float4)*total_recieved);
    float4* h_pos = (float4*)malloc(sizeof(float4)*params.n_particles);
    float4* h_vel = (float4*)malloc(sizeof(float4)*params.n_particles);

    cudaCall(cudaMemcpy,h_pos,mem.d_pos,sizeof(float4)*params.n_particles,cudaMemcpyDeviceToHost);
    cudaCall(cudaMemcpy,h_vel,mem.d_vel,sizeof(float4)*params.n_particles,cudaMemcpyDeviceToHost);
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
    idx = remaining;
    for (int i = 0; i < total_recieved; i++){
        float4 tmp_pos = pos_to_transfer[i];
        float4 tmp_vel = vel_to_transfer[i];
        if ((idx) >= params.n_particles){
            printf("PARTICLE OVERFLOW!!!!!!!!!!\n");
        }
        float particle_id = tmp_pos.w;
        bool add = true;
        for (int j = 0; j < params.n_particles; j++){
            if (h_pos[j].w == particle_id){
                add = false;
                printf("Found duplicate particle!!!\n");
                break;
            }
        }
        if (add){
            h_pos[idx] = tmp_pos;
            h_vel[idx] = tmp_vel;
            idx++;
        }
    }

    cudaCall(cudaMemcpy,mem.d_pos,h_pos,sizeof(float4)*params.n_particles,cudaMemcpyHostToDevice);
    cudaCall(cudaMemcpy,mem.d_vel,h_vel,sizeof(float4)*params.n_particles,cudaMemcpyHostToDevice);

    free(h_pos);
    free(h_vel);
    free(pos_to_transfer);
    free(vel_to_transfer);
}