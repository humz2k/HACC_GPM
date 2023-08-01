#include "../ic_kernels.hpp"

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
    
    double logy = logy1 + ((logy2 - logy1)/(logx2 - logx1)) * (logx - logx1);
    double y = exp(logy) * (((double)(ng*ng*ng))/(rl*rl*rl));
    if (left_bin == 0){
        y = in[right_bin] * (((double)(ng*ng*ng))/(rl*rl*rl));
    }

    return y;

}

__global__ void interpolatePowerSpectrum(hostFFT_t* out, double* in, int nbins, double k_delta, double k_min, double rl, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);

    out[idx] = do_pk_interpolation(idx3d,in,k_delta,k_min,rl,ng);

}

__global__ void interpolatePowerSpectrum(hostFFT_t* out, double* in, int nbins, double k_delta, double k_min, double rl, int ng, int nlocal, int world_rank, int3 local_grid_size, int3 local_coords, int3 dims){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= nlocal)return;

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);

    out[idx] = do_pk_interpolation(idx3d,in,k_delta,k_min,rl,ng);

}

CPUTimer_t launch_interpolate_pk(HACCGPM::CosmoClass& cosmo, hostFFT_t* d_pkScale, int ng, double rl, int numBlocks, int blockSize, int calls){
    getIndent(calls);

    CPUTimer_t start = CPUTimer();

    double* h_ipk;
    int ipk_bins;
    double ipk_delta;
    double ipk_max;
    double ipk_min;
    cosmo.read_ipk(&h_ipk,&ipk_bins,&ipk_delta,&ipk_max,&ipk_min,calls+1);
    double* d_ipk; cudaCall(cudaMalloc,&d_ipk,sizeof(double)*ipk_bins);
    cudaCall(cudaMemcpy, d_ipk, h_ipk, sizeof(double)*ipk_bins, cudaMemcpyHostToDevice);

    double maxK = ((ng/2)*2*M_PI)/rl;
    maxK = sqrt(maxK*maxK*maxK);
    printf("%s      maxK = %g\n",indent,maxK);
    if (maxK > ipk_max){
        printf("%s      input ipk only goes to %g\n",indent,ipk_max);
        exit(1);
    }

    InvokeGPUKernel(interpolatePowerSpectrum,numBlocks,blockSize,d_pkScale,d_ipk,ipk_bins,ipk_delta,ipk_min,rl,ng);

    free(h_ipk);
    cudaCall(cudaFree,d_ipk);

    CPUTimer_t end = CPUTimer();

    return end - start;
}

CPUTimer_t launch_interpolate_pk(HACCGPM::CosmoClass& cosmo, hostFFT_t* d_pkScale, int ng, double rl, int nlocal, int3 local_grid_size, int3 local_coords, int3 dims, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);

    CPUTimer_t start = CPUTimer();

    double* h_ipk;
    int ipk_bins;
    double ipk_delta;
    double ipk_max;
    double ipk_min;
    cosmo.read_ipk(&h_ipk,&ipk_bins,&ipk_delta,&ipk_max,&ipk_min,calls+1);
    double* d_ipk; cudaCall(cudaMalloc,&d_ipk,sizeof(double)*ipk_bins);
    cudaCall(cudaMemcpy, d_ipk, h_ipk, sizeof(double)*ipk_bins, cudaMemcpyHostToDevice);

    double maxK = ((ng/2)*2*M_PI)/rl;
    maxK = sqrt(maxK*maxK*maxK);
    if(world_rank == 0)printf("%s      maxK = %g\n",indent,maxK);
    if (maxK > ipk_max){
        if(world_rank == 0)printf("%s      input ipk only goes to %g\n",indent,ipk_max);
        exit(1);
    }
    
    InvokeGPUKernelParallel(interpolatePowerSpectrum,numBlocks,blockSize,d_pkScale,d_ipk,ipk_bins,ipk_delta,ipk_min,rl,ng,nlocal,world_rank,local_grid_size,local_coords,dims);

    free(h_ipk);
    cudaCall(cudaFree,d_ipk);

    CPUTimer_t end = CPUTimer();

    return end - start;
}