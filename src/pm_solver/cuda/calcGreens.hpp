#include "../kernels.hpp"

__device__ __forceinline__ double3 cos(float3 kmodes){
    double3 out;
    out.x = cos(kmodes.x);
    out.y = cos(kmodes.y);
    out.z = cos(kmodes.z);
    return out;
}

__forceinline__ __device__ double calcGreens(int3 idx3d, int ng, int np){

    if ((idx3d.x == 0) && (idx3d.y == 0) && (idx3d.z == 0))return 0.0;

    float d = ((2*M_PI)/(((float)(ng))));

    double3 c = cos(idx3d * d);

    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    double coeff = 0.5 / (ng*ng*ng);

    double out = coeff / (c.x + c.y + c.z - 3.0);

    return out;

}