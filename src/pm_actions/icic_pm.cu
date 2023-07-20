#include "pm_kernels.hpp"

__global__ void ICICKernel(float4* __restrict d_vel, const float4* __restrict d_grad, const float4* __restrict my_pos, double deltaT, double fscal, int ng){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    float4 my_particle = __ldg(&my_pos[idx]);
    float3 my_deltaV = make_float3(0.0,0.0,0.0);
    int i = my_particle.x;
    int j = my_particle.y;
    int k = my_particle.z;

    float diffx = (my_particle.x - (float)i);
    float diffy = (my_particle.y - (float)j);
    float diffz = (my_particle.z - (float)k);

    for (int x = 0; x < 2; x++){
        for (int y = 0; y < 2; y++){
            for (int z = 0; z < 2; z++){
                int nx = (i + x)%ng;
                int ny = (j + y)%ng;
                int nz = (k + z)%ng;
                int indx = (nx)*ng*ng + (ny)*ng + nz;

                float dx = diffx;
                if (x == 0){
                    dx = 1 - dx;
                }
                float dy = diffy;
                if (y == 0){
                    dy = 1 - dy;
                }
                float dz = diffz;
                if (z == 0){
                    dz = 1 - dz;
                }

                float4 grad = __ldg(&d_grad[indx]);

                float mul = dx*dy*dz * deltaT * (fscal);//* (1.0f/((double)(ng*ng*ng)));// (1.0f/((double)(ng*ng*ng)));// * deltaT * fscal * (1.0f/((double)(ng*ng*ng)));
                my_deltaV.x += mul*grad.x;
                my_deltaV.y += mul*grad.y;
                my_deltaV.z += mul*grad.z;

                //atomicAdd(&grid[indx].x,(double)mul);
            }
        }
    }

    float4 my_vel = __ldg(&d_vel[idx]);
    my_vel.x += my_deltaV.x;
    my_vel.y += my_deltaV.y;
    my_vel.z += my_deltaV.z;

    d_vel[idx] = my_vel;

}