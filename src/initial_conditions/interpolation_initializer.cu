#include "ic_kernels.hpp"

__forceinline__ __device__ double do_pk_interpolation(int3 idx3d, double* in, double k_delta, double k_min, double rl, int ng){
    
    if ((idx3d.x == 0) && (idx3d.y == 0) && (idx3d.z == 0))return 0.0;

    double d = (2*M_PI)/rl;

    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    double my_k = sqrt(kmodes.x*kmodes.x + kmodes.y*kmodes.y + kmodes.z*kmodes.z);

    int left_bin = (int)(my_k / k_delta);
    int right_bin = left_bin + 1;

    double logy1 = log(in[left_bin]);
    double logx1 = log(k_delta * (double)left_bin);
    double logy2 = log(in[right_bin]);
    double logx2 = log(k_delta * (double)right_bin);
    double logx = log(my_k);
    //double frac = 1.0f/abs(logx1 - logx2);
    double logy = logy1 + ((logy2 - logy1)/(logx2 - logx1)) * (logx - logx1);
    double y = exp(logy) * (((double)(ng*ng*ng))/(rl*rl*rl));
    if (left_bin == 0){
        //printf("%d -> %d: %g -> %g\n",left_bin,right_bin,in[left_bin] * (((double)(ng*ng*ng))/(rl*rl*rl)),in[right_bin] * (((double)(ng*ng*ng))/(rl*rl*rl)));
        y = in[right_bin] * (((double)(ng*ng*ng))/(rl*rl*rl));
    }

    return 0;

}

__global__ void interpolatePowerSpectrum(hostFFT_t* out, double* in, int nbins, double k_delta, double k_min, double rl, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx == 0){
        out[idx] = 0;
        return;
    }

    double d = (2*M_PI)/rl;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    double my_k = sqrt(kmodes.x*kmodes.x + kmodes.y*kmodes.y + kmodes.z*kmodes.z);

    int left_bin = (int)(my_k / k_delta);
    int right_bin = left_bin + 1;

    double logy1 = log(in[left_bin]);
    double logx1 = log(k_delta * (double)left_bin);
    double logy2 = log(in[right_bin]);
    double logx2 = log(k_delta * (double)right_bin);
    double logx = log(my_k);
    //double frac = 1.0f/abs(logx1 - logx2);
    double logy = logy1 + ((logy2 - logy1)/(logx2 - logx1)) * (logx - logx1);
    double y = exp(logy) * (((double)(ng*ng*ng))/(rl*rl*rl));
    if (left_bin == 0){
        //printf("%d -> %d: %g -> %g\n",left_bin,right_bin,in[left_bin] * (((double)(ng*ng*ng))/(rl*rl*rl)),in[right_bin] * (((double)(ng*ng*ng))/(rl*rl*rl)));
        y = in[right_bin] * (((double)(ng*ng*ng))/(rl*rl*rl));
    }
    //printf("%g: %g > %g > %g\n",my_k,exp(logy1)*(((double)(ng*ng*ng))/(rl*rl*rl)),y,exp(logy2)*(((double)(ng*ng*ng))/(rl*rl*rl)));
    //printf("x1,y1=%g,%g, x2,y2=%g,%g\n",logx1,logy1,logx2,logy2);
    out[idx] = (y);

}

__global__ void interpolatePowerSpectrum(hostFFT_t* out, double* in, int nbins, double k_delta, double k_min, double rl, int ng, int nlocal, int world_rank, int3 local_grid_size, int3 local_coords, int3 dims){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= nlocal)return;

    if (idx == 0){
        out[idx] = 0;
        return;
    }

    double d = (2*M_PI)/rl;

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    double my_k = sqrt(kmodes.x*kmodes.x + kmodes.y*kmodes.y + kmodes.z*kmodes.z);

    int left_bin = (int)(my_k / k_delta);
    int right_bin = left_bin + 1;

    double logy1 = log(in[left_bin]);
    double logx1 = log(k_delta * (double)left_bin);
    double logy2 = log(in[right_bin]);
    double logx2 = log(k_delta * (double)right_bin);
    double logx = log(my_k);
    //double frac = 1.0f/abs(logx1 - logx2);
    double logy = logy1 + ((logy2 - logy1)/(logx2 - logx1)) * (logx - logx1);
    double y = exp(logy) * (((double)(ng*ng*ng))/(rl*rl*rl));
    if (left_bin == 0){
        //printf("%d -> %d: %g -> %g\n",left_bin,right_bin,in[left_bin] * (((double)(ng*ng*ng))/(rl*rl*rl)),in[right_bin] * (((double)(ng*ng*ng))/(rl*rl*rl)));
        y = in[right_bin] * (((double)(ng*ng*ng))/(rl*rl*rl));
    }
    //printf("%g: %g > %g > %g\n",my_k,exp(logy1)*(((double)(ng*ng*ng))/(rl*rl*rl)),y,exp(logy2)*(((double)(ng*ng*ng))/(rl*rl*rl)));
    //printf("x1,y1=%g,%g, x2,y2=%g,%g\n",logx1,logy1,logx2,logy2);
    out[idx] = (y);

}