#include <cuda_runtime.h>
#include <math.h>

#define VEC_OP_PREAMBLE __forceinline__ __host__ __device__

#define OP_float3_float(a,op,s) make_float3(a.x op s, a.y op s, a.z op s)
#define OP_int3_int(a,op,s) make_int3(a.x op s, a.y op s, a.z op s)
#define OP_int_int3(s,op,a) make_int3(s op a.x, s op a.y, s op a.z)
#define OP_float_float3(s,op,a) make_float3(s op a.x, s op a.y, s op a.z)
#define OP_float4_float(a,op,s) make_float4(a.x op s, a.y op s, a.z op s, a.w op s)
#define OP_float_float4(s,op,a) make_float4(s op a.x, s op a.y, s op a.z, s op a.w)
#define OP_float4_float4(a,op,s) make_float4(a.x op s.x, a.y op s.y, a.z op s.z, a.w op s.w)
#define OP_float4_float3(a,op,s) make_float4(a.x op s.x, a.y op s.y, a.z op s.z, a.w)
#define OP_float3_float4(a,op,s) make_float4(a.x op s.x, a.y op s.y, a.z op s.z, 0 op s.w)
#define OP_float3_float3(a,op,s) make_float3(a.x op s.x, a.y op s.y, a.z op s.z)

#define OP_vec(op) VEC_OP_PREAMBLE float3 operator op(float3 a, float s){return OP_float3_float(a,op,s);}\
VEC_OP_PREAMBLE float3 operator op(float s, float3 a){return OP_float_float3(s,op,a);}\
VEC_OP_PREAMBLE float3 operator op(int3 a, float s){return OP_float3_float(a,op,s);}\
VEC_OP_PREAMBLE int3 operator op(int3 a, int s){return OP_int3_int(a,op,s);}\
VEC_OP_PREAMBLE int3 operator op(int s, int3 a){return OP_int_int3(s,op,a);}\
VEC_OP_PREAMBLE float3 operator op(float s, int3 a){return OP_float_float3(s,op,a);}\
VEC_OP_PREAMBLE float4 operator op(float4 a, float s){return OP_float4_float(a,op,s);}\
VEC_OP_PREAMBLE float4 operator op(float s, float4 a){return OP_float_float4(s,op,a);}\
VEC_OP_PREAMBLE float4 operator op(float4 a, float4 s){return OP_float4_float4(a,op,s);}\
VEC_OP_PREAMBLE float4 operator op(float4 a, float3 s){return OP_float4_float3(a,op,s);}\
VEC_OP_PREAMBLE float4 operator op(float3 a, float4 s){return OP_float3_float4(a,op,s);}\
VEC_OP_PREAMBLE float4 operator op(float4 a, int3 s){return OP_float4_float3(a,op,s);}\
VEC_OP_PREAMBLE float4 operator op(int3 a, float4 s){return OP_float3_float4(a,op,s);}\
VEC_OP_PREAMBLE float3 operator op(float3 a, float3 s){return OP_float3_float3(a,op,s);}\
VEC_OP_PREAMBLE float3 operator op(float3 a, int3 s){return OP_float3_float3(a,op,s);}\
VEC_OP_PREAMBLE float3 operator op(int3 a, float3 s){return OP_float3_float3(a,op,s);}

OP_vec(+);
OP_vec(-);
OP_vec(*);
OP_vec(/);

VEC_OP_PREAMBLE float3 fmod(float3 a, float s){
    return make_float3(fmod(a.x,s),fmod(a.y,s),fmod(a.z,s));
}