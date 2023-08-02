#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <cstring>
#include "vector_operators.hpp"

#ifndef NOPYTHON
#include "pycosmotools.hpp"
#endif

#define USECPSEC 1000000ULL

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

typedef unsigned long long CPUTimer_t;

inline unsigned long long CPUTimer(unsigned long long start=0){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

#define deviceFFT_t cufftDoubleComplex
#define floatFFT_t cufftComplex
#define hostFFT_t double

#define getIndent(calls) char indent[50] = ""; for (int i = 0; i < (calls*6); i++){strcat(indent," ");}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   } else{
   }
}

inline __device__ float3 __ldg(const float3* i){
    return *i;
}

#define cudaCall(func,...) if (func(__VA_ARGS__) != cudaSuccess)printf("Error >> %s\n", cudaGetErrorString(cudaGetLastError()));cudaDeviceSynchronize()

//#define VerboseCuda

#ifdef VerboseCuda

#define InvokeGPUKernel(func,numBlocks,blockSize,...) ({printf("%s   Invoked %s\n",indent,TOSTRING(func));CPUTimer_t start = CPUTimer();func<<<numBlocks,blockSize>>>(__VA_ARGS__);gpuErrchk( cudaPeekAtLastError() );gpuErrchk( cudaDeviceSynchronize() );CPUTimer_t end = CPUTimer();CPUTimer_t t = end-start;printf("%s      %s took %llu us\n",indent,TOSTRING(func),t); t;})
#define InvokeGPUKernelParallel(func,numBlocks,blockSize,...) ({if(world_rank == 0)printf("%s   Invoked %s\n",indent,TOSTRING(func));CPUTimer_t start = CPUTimer();func<<<numBlocks,blockSize>>>(__VA_ARGS__);gpuErrchk( cudaPeekAtLastError() );gpuErrchk( cudaDeviceSynchronize() );CPUTimer_t end = CPUTimer();CPUTimer_t t = end-start;if(world_rank == 0)printf("%s      %s took %llu us\n",indent,TOSTRING(func),t); t;})

#else

#define InvokeGPUKernel(func,numBlocks,blockSize,...) ({CPUTimer_t start = CPUTimer();func<<<numBlocks,blockSize>>>(__VA_ARGS__);gpuErrchk( cudaPeekAtLastError() );gpuErrchk( cudaDeviceSynchronize() );CPUTimer_t end = CPUTimer();CPUTimer_t t = end-start; printf("%s: %llu us\n",TOSTRING(func),t); t;})
#define InvokeGPUKernelParallel(func,numBlocks,blockSize,...) ({CPUTimer_t start = CPUTimer();func<<<numBlocks,blockSize>>>(__VA_ARGS__);gpuErrchk( cudaPeekAtLastError() );gpuErrchk( cudaDeviceSynchronize() );CPUTimer_t end = CPUTimer();CPUTimer_t t = end-start; t;})

#endif

#define hostFFT_t double
#define deviceFFT_t cufftDoubleComplex

#define USE_SINGLE_FFT
#define USE_SINGLE_GREENS
#define USE_FLOAT3
#define USE_ONE_GRID
//#define USE_TEMP_GRID
//#define USE_GREENS_CACHE

//#define USE_HALF_PRECISION

#ifdef USE_FLOAT3
#define particle_t float3
#else
#define particle_t float4
#endif

#ifdef USE_SINGLE_GREENS
#define greens_t float
#else
#define greens_t hostFFT_t
#endif

#ifdef USE_SINGLE_FFT
#define grid_t floatFFT_t
#else
#define grid_t deviceFFT_t
#endif

#ifdef USE_HALF_PRECISION
#include <cuda_fp16.h>

class half4{
    half x;
    half y;
    half z;
    unsigned short w;
};

#endif

#define MAX_STEPS 1000

#define HostMalloc(ptr,sz) cudaCall(cudaMallocHost,ptr,sz)
#define HostFree(ptr) cudaCall(cudaFreeHost,ptr)

//__forceinline__ __host__ __device__ float3 operator*(int3 a, float s)
//{
//    return make_float3(a.x * s, a.y * s, a.z * s);
//}

namespace HACCGPM{

    struct Params{
        char fname[300];
        int ng;
        int np;
        double rl;
        int blockSize;
        int seed;
        double z_ini;
        double z_fin;
        int nsteps;
        double deltaT;
        char prefix[300];
        double m_hubble;
        double m_omega_cdm;
        double m_omega_nu;
        double m_deut;
        double m_ss8;
        double m_ns;
        double m_w_de;
        double m_wa_de;
        double m_Tcmb;
        double m_neff_massless;
        double m_neff_massive;
        double m_omega_baryon;
        double m_omega_cb;
        double m_omega_matter;
        double m_omega_radiation;
        double m_f_nu_massless;
        double m_f_nu_massive;
        bool pks[MAX_STEPS];
        bool dumps[MAX_STEPS];
        bool analysis[MAX_STEPS];
        char analysis_dir[300];
        char analysis_py[300];
        char ipk[300];
        bool do_analysis;
        int lastStep;
        int pkFolds;
        int nlocal;
        float frac;
        int world_rank;
        int world_size;
        int local_grid_start[3];
        int local_grid_size[3];
        int grid_dims[3];
        int grid_coords[3];
        int n_particles;
        bool dump_init;
        bool dump_final;
        double ol;
        int overload;
        int3 local_grid_start_vec;
        int3 local_grid_size_vec;
        int3 grid_dims_vec;
        int3 grid_coords_vec;
        int pk_bins;
    };

    Params read_params(const char* fname);

    class Timestepper{
        public:
            Params params;
            double aa;
            double z;
            double deltaT;
            double adot;
            double fscal;
            int world_rank;

            Timestepper(Params iparams);

            void setInitialA(double a);

            void setInitialZ(double z);

            void advanceHalfStep();

            void reverseHalfStep();
    };

    class CosmoClass{
        public:
            //HACCGPM::Params& params;
            float Omega_m;
            float Omega_cdm;
            float Omega_bar;
            float Omega_cb;
            float Omega_nu;
            float f_nu_massless;
            float f_nu_massive;
            float Omega_r;
            float h;
            float w_de;
            float wa_de;
            char ipk[300];
            //float Omega_cb;
            //float f_nu_massless;
            //float Omega_r;
            //float Omega_m;
            //float w_de;
            //float wa_de;
            //float Omega_nu;
            //float f_nu_massive;

            void read_ipk(double** out, int* nbins, double* k_delta, double* k_max, double* k_min, int calls = 0);

            CosmoClass(){};
            CosmoClass(HACCGPM::Params& params_);
            ~CosmoClass(){};
            void GrowthFactor(float z, float* gf, float* g_dot);
            void odesolve(float* ystart, int nvar, 
                                float x1, float x2, float eps, float h1,
                                void (HACCGPM::CosmoClass::*derivs)(float, float*, float*), bool print_stat);
            void rkqs(float* y, float* dydx, int n, float* x, float htry,
                                float eps, float* yscal, float* hdid, float* hnext,
                                int* feval, void (HACCGPM::CosmoClass::*derivs)(float, float*, float*));
            void rkck(float* y, float* dydx, int n, float x, float h,
		      float* yout, float* yerr,
		      void (HACCGPM::CosmoClass::*derivs)(float, float* , float* ));
            
            void growths(float a, float* y, float* dydx);

            float da_dtau(float a, float OmM, float OmL);
            float da_dtau__3(float a, float OmM, float OmL);
            float int_1_da_dtau_3(float a, float OmM, float OmL, int bins=10);
            float delta(float a, float OmM, float OmL);
            float dotDelta(float a, float OmM, float OmL, float h = 0.001);

            float z2a(float z);
            float a2z(float a);
            void get_delta_and_dotDelta(float z, float z1, float* d, float* d_dot);
            void get_delta_and_dotDelta(float z, float z1, double* d, double* d_dot);

            float Omega_nu_massive(float a);
    };

    __device__ __forceinline__ float3 get_kmodes(int3 idx3d, int ng, double d){
            float l = idx3d.x;
            if (idx3d.x > ((ng/2)-1)){
                l = -(ng - idx3d.x);
            }
            l *= d;

            float m = idx3d.y;
            if (idx3d.y > ((ng/2)-1)){
                m = -(ng - idx3d.y);
            }
            m *= d;

            float n = idx3d.z;
            if (idx3d.z > ((ng/2)-1)){
                n = -(ng - idx3d.z);
            }
            n *= d;
            return make_float3(l,m,n);
        }

    namespace parallel{
        class MemoryManager{
            public:
                int world_rank;
                float4* d_pos;
                float4* d_vel;
                float4* d_grad;
                hostFFT_t* d_greens;
                deviceFFT_t* d_grid;
                deviceFFT_t* d_x;
                deviceFFT_t* d_y;
                deviceFFT_t* d_z;
                float* d_tempgrid;
                //deviceFFT_t* d_grid2;

                MemoryManager(HACCGPM::Params params);

                ~MemoryManager();
        };

        class GridExchange{

            private:
                int3 local_coords;
                int3 local_grid_size;
                int3 dims;
                int ng;
                int world_size;
                int world_rank;
                int overload;
                int blockSize;

            public:
                GridExchange();
                GridExchange(int3 local_coords_, int3 local_grid_size_, int3 dims_, int ng_, int world_size_, int world_rank_, int overload_, int blockSize_);

                void resolve(float* grid, int calls = 0);

                void fill(float4* grad, int calls = 0);
        };

        __device__ __forceinline__ int3 get_local_index(int idx, int nx, int ny, int nz){
            int3 out;
            out.x = idx/(ny*nz);
            out.y = (idx - (out.x*ny*nz))/nz;
            out.z = (idx - (out.x*ny*nz)) - out.y*nz;
            return out;
        }

        __device__ __forceinline__ int3 get_global_index(int idx, int ng, int3 local_dims, int3 local_coords){
            int3 out;
            int3 local = get_local_index(idx,local_dims.x,local_dims.y,local_dims.z);
            out.x = local.x + local_coords.x * local_dims.x;
            out.y = local.y + local_coords.y * local_dims.y;
            out.z = local.z + local_coords.z * local_dims.z;
            return out;
        }

        void init_swfft(HACCGPM::Params& params);
        void finalize_swfft();
        void forward_fft(deviceFFT_t* d_grid, int ng, int calls = 0);
        void backward_fft(deviceFFT_t* d_grid, int ng, int calls = 0);

        void timing_stats(CPUTimer_t t, CPUTimer_t* mint, CPUTimer_t* maxt, CPUTimer_t* meant);
        void printTimingStats(const char *preamble, double dt);

        CPUTimer_t LoadIntoBuffers(float4* h_swap, int* n_swaps, int* h_starts, float4* d_pos, float4* d_vel, int nlocal, int3 local_grid_size, int3 local_coords, int3 dims, int n_particles, int ng, int blockSize, int world_rank, int world_size, int calls = 0);

        void TransferParticles(HACCGPM::Params& params,HACCGPM::parallel::MemoryManager& mem, int calls = 0);
        CPUTimer_t insertParticles(float4* d_pos, float4* d_vel, float4* h_swap, int n_new, int n_particles, int blockSize, int world_rank, int calls = 0);

        void GenerateDisplacementIC(HACCGPM::parallel::MemoryManager& mem, HACCGPM::CosmoClass& cosmo, HACCGPM::Params& params, HACCGPM::Timestepper& ts, int calls = 0);

        void CIC(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager&mem, int calls = 0);

        void CIC(deviceFFT_t* d_grid, float* d_extragrid, float4* d_pos, int ng, int n_particles, int3 local_grid_size, int3 local_coords, int3 dims, int blockSize, int world_rank, int world_size, int overload, int calls = 0);

        void GetPowerSpectrum(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager& mem, int nbins, const char* fname, int calls = 0);

        void sendPower(int* binCounts, double* binVals, int nbins, int world_rank, int world_size, int calls = 0);

        void InitGreens(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager& mem, int calls = 0);

        void UpdatePositions(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager& mem, HACCGPM::Timestepper ts, float frac, int calls = 0);

        int UpdatePositions(float4* d_pos, float4* d_vel, HACCGPM::Timestepper ts, float frac, int ng, int n_particles, int3 local_grid_size, int overload, int blockSize, int world_rank, int calls = 0);

        void UpdateVelocities(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager& mem, HACCGPM::Timestepper ts, int calls = 0);

        void UpdateVelocities(float4* d_vel, float4* d_grad, float4* d_pos, HACCGPM::Timestepper ts, int n_particles, int ng, int overload, int3 local_grid_size, int blockSize, int world_rank, int calls = 0);

        void SolveGradient(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager& mem, int calls = 0);

        void SolveGradient(float4* d_grad, deviceFFT_t* d_rho, hostFFT_t* d_greens, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, int ng, int n_particles, int nlocal, int3 local_grid_size, int3 local_coords, int3 dims, int world_rank, int world_size, int overload, int blockSize, int calls = 0);

        void printTransferTimes(int world_rank);

        void printTransferBytes(int world_rank);

        void printFFTStats(int world_rank);

        void printPATimes(int world_rank);

        void printTimers(CPUTimer_t init, CPUTimer_t total, int world_rank);

        void printGridExchangeTimes(int world_rank);
        void printGridExchangeBytes(int world_rank);
    }

    namespace serial{

        class MemoryManager{
            public:

                #ifdef USE_HALF_PRECISION
                half4* d_pos;
                #else
                particle_t* d_pos;
                #endif

                particle_t* d_vel;
                float4* d_grad;

                #ifdef USE_GREENS_CACHE
                greens_t* d_greens;
                #endif

                grid_t* d_grid;

                #ifdef USE_TEMP_GRID
                float* d_tempgrid;
                #endif

                #ifndef USE_ONE_GRID
                grid_t* d_x;
                grid_t* d_y;
                grid_t* d_z;
                #endif

                int* d_binCounts;
                float* d_binVals;

                MemoryManager(HACCGPM::Params params);

                ~MemoryManager();
        };

        __device__ __forceinline__ int3 get_index(int idx, int ng){
            int3 out;
            out.x = idx/(ng*ng);
            out.y = (idx - (out.x*ng*ng))/ng;
            out.z = (idx - (out.x*ng*ng)) - out.y*ng;
            return out;
        }

        void UpdatePositions(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, HACCGPM::Timestepper ts, float frac, int calls = 0);

        template<class T>
        void UpdatePositions(T* d_pos, T* d_vel, HACCGPM::Timestepper ts, float frac, int ng, int np, int blockSize, int calls = 0);

        void UpdateVelocities(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, HACCGPM::Timestepper ts, int calls = 0);

        template<class T>
        void UpdateVelocities(T* d_vel, float4* d_grad, T* d_pos, HACCGPM::Timestepper ts, int ng, int np, int blockSize, int calls = 0);

        template<class T1, class T2>
        void CIC(T1* d_grid, T2* d_pos, int ng, int np, int blockSize, int calls);

        template<class T1, class T2>
        void CIC(T1* d_grid, float* d_temp, T2* d_pos, int ng, int np, int blockSize, int calls);

        void CIC(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, int calls = 0);

        double get_fscal(double m_aa, double m_adot, double m_omega_cb);

        double get_adot(double m_aa, double m_w, double m_wa, double m_omega_cb, double m_omega_radiation, double m_f_nu_massless, double m_f_nu_massive, double m_omega_matter, double m_omega_nu);

        void writeOutput(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, char* fname, int calls = 0);

        template<class T>
        void writeOutput(char* fname, T* d_pos, T* d_vel, int ng, int np, int calls = 0);

        void GetPowerSpectrum(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, const char* fname, int calls = 0);

        void GenerateDisplacementIC(HACCGPM::serial::MemoryManager& mem, HACCGPM::CosmoClass& cosmo, HACCGPM::Params& params, HACCGPM::Timestepper& ts, int calls = 0);

        void Solve(deviceFFT_t* d_rho, hostFFT_t* d_greens, int ng, int blockSize, int calls = 0);

        void SolveGradient(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, int calls = 0);

        template<class T1, class T2>
        void SolveGradient(float4* d_grad, T1* d_rho, T2* d_greens, T1* d_x, T1* d_y, T1* d_z, int ng, int blockSize, int calls = 0);

        template<class T1, class T2>
        void SolveGradient(float4* d_grad, T1* d_rho, T2* d_greens, int ng, int blockSize, int calls = 0);

        template<class T>
        void SolveGradient(float4* d_grad, T* d_rho, T* d_x, T* d_y, T* d_z, int ng, int np, int blockSize, int calls = 0);

        template<class T>
        void SolveGradient(float4* d_grad, T* d_rho, int ng, int np, int blockSize, int calls = 0);

        void InitGreens(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, int calls = 0);

        void forward_fft(deviceFFT_t* data, deviceFFT_t* out, int ng, int calls = 0);

        void forward_fft(deviceFFT_t* data, int ng, int calls = 0);

        void forward_fft(floatFFT_t* d_grid, int ng, int calls = 0);
        void backward_fft(floatFFT_t* d_grid, int ng, int calls = 0);

        void backward_fft(deviceFFT_t* data, deviceFFT_t* out, int ng, int calls = 0);

        void backward_fft(deviceFFT_t* data, int ng, int calls = 0);

        void fft_cache_plan(int ng);

        void printCICTimes();

        void printFFTTimes();

        void printPowerTimes();

        void printOutputTimes();

        void printTimers(CPUTimer_t init, CPUTimer_t total);

        #ifndef NOPYTHON
        void PyAnalysis(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, HACCGPM::Timestepper& ts, PyCosmoTools& pytools, int step);
        #endif
    }
}