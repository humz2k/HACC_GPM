#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <cstring>
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

#define cudaCall(func,...) if (func(__VA_ARGS__) != cudaSuccess)printf("Error >> %s\n", cudaGetErrorString(cudaGetLastError()));cudaDeviceSynchronize()

#define InvokeGPUKernel(func,numBlocks,blockSize,...) ({printf("%s   Invoked %s\n",indent,TOSTRING(func));CPUTimer_t start = CPUTimer();func<<<numBlocks,blockSize>>>(__VA_ARGS__);gpuErrchk( cudaPeekAtLastError() );gpuErrchk( cudaDeviceSynchronize() );CPUTimer_t end = CPUTimer();CPUTimer_t t = end-start;printf("%s      %s took %llu us\n",indent,TOSTRING(func),t); t;})

#define InvokeGPUKernelParallel(func,numBlocks,blockSize,...) ({if(world_rank == 0)printf("%s   Invoked %s\n",indent,TOSTRING(func));CPUTimer_t start = CPUTimer();func<<<numBlocks,blockSize>>>(__VA_ARGS__);gpuErrchk( cudaPeekAtLastError() );gpuErrchk( cudaDeviceSynchronize() );CPUTimer_t end = CPUTimer();CPUTimer_t t = end-start;if(world_rank == 0)printf("%s      %s took %llu us\n",indent,TOSTRING(func),t); t;})

#define hostFFT_t double
#define deviceFFT_t cufftDoubleComplex

#define MAX_STEPS 1000

namespace HACCGPM{

    struct Params{
        int ng;
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

    namespace parallel{
        class MemoryManager{
            public:
                int world_rank;
                float4* d_pos;
                float4* d_vel;
                float4* d_grad;
                hostFFT_t* d_greens;
                deviceFFT_t* d_grid;
                float* d_tempgrid;
                //deviceFFT_t* d_grid2;

                MemoryManager(HACCGPM::Params params);

                ~MemoryManager();
        };

        __device__ __forceinline__ int3 get_local_index(int idx, int nx, int ny, int nz){
            int3 out;
            out.x = idx/(ny*nz);
            out.y = (idx - (out.x*ny*nz))/nz;
            out.z = (idx - (out.x*ny*nz)) - out.y*nz;
            return out;
        }

        __device__ __forceinline__ int3 get_global_index(int idx, int ng, int nlocal, int world_rank){
            int3 out;
            idx += nlocal * world_rank;
            out.x = idx/(ng*ng);
            out.y = (idx - (out.x*ng*ng))/ng;
            out.z = (idx - (out.x*ng*ng)) - out.y*ng;
            return out;
        }

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

        void init_swfft(HACCGPM::Params& params);
        void finalize_swfft();
        void forward_fft(deviceFFT_t* d_grid, int ng, int calls = 0);
        void backward_fft(deviceFFT_t* d_grid, int ng, int calls = 0);

        void timing_stats(CPUTimer_t t, CPUTimer_t* mint, CPUTimer_t* maxt, CPUTimer_t* meant);

        CPUTimer_t initLoadIntoBuffers(float4** swap_pos, float4** swap_vel, int* n_swaps, float4* d_pos, float4* d_vel, int nlocal, int3 local_grid_size, int n_particles, int blockSize, int world_rank, int calls = 0);

        void initTransferParticles(HACCGPM::Params& params,HACCGPM::parallel::MemoryManager& mem, int calls = 0);
        CPUTimer_t insertParticles(float4* d_pos, float4* d_vel, float4* new_pos, float4* new_vel, int n_new, int remaining, int n_particles, int blockSize, int world_rank, int calls = 0);

        CPUTimer_t loadGridBuffers(float* d_extragrid, float* h_transfer, int3 local_grid_size, int blockSize, int world_rank, int calls = 0);

        void GenerateDisplacementIC(const char* params_file, HACCGPM::parallel::MemoryManager* mem, int ng, double rl, double z, double deltaT, double fscal, int seed, int blockSize, int world_rank, int world_size, int nlocal, int* local_grid_size, int calls = 0);
    
        void CIC(deviceFFT_t* d_grid, float* d_extragrid, float4* d_pos, int ng, int n_particles, int* local_grid_size, int blockSize, int world_rank, int world_size, int calls = 0);

        void gridExchange(float* d_extragrid, int3 local_grid_size, int world_rank, int world_size, int blockSize, int calls = 0);

        void printTransferTimes(int world_rank);
        void printCICTimes(int world_rank);

        void printTransferBytes(int world_rank);

        void printFFTStats(int world_rank);
    }

    namespace serial{

        class MemoryManager{
            public:
                float4* d_pos;
                float4* d_vel;
                float4* d_grad;
                hostFFT_t* d_greens;
                deviceFFT_t* d_grid;
                float* d_tempgrid;

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

        void UpdatePositions(float4* d_pos, float4* d_vel, HACCGPM::Timestepper ts, float frac, int ng, int blockSize, int calls = 0);

        void UpdateVelocities(float4* d_vel, float4* d_grad, float4* d_pos, HACCGPM::Timestepper ts, int ng, int blockSize, int calls = 0);

        void CIC(deviceFFT_t* d_grid, float4* d_pos, int ng, int blockSize, int calls = 0);

        void CIC(float* d_grid, float4* d_pos, int ng, int blockSize, int calls = 0);

        void CIC(deviceFFT_t* d_grid, float* d_temp, float4* d_pos, int ng, int blockSize, int calls = 0);

        double get_fscal(double m_aa, double m_adot, double m_omega_cb);

        double get_adot(double m_aa, double m_w, double m_wa, double m_omega_cb, double m_omega_radiation, double m_f_nu_massless, double m_f_nu_massive, double m_omega_matter, double m_omega_nu);

        void writeOutput(char* fname, float4* d_pos, float4* d_vel, int ng, int calls = 0);

        void GetPowerSpectrum(float4* d_pos, deviceFFT_t* d_grid, int ng, double rl, int nbins, const char* fname, int nfolds, int blockSize, int calls = 0);

        void GetPowerSpectrum(float4* d_pos, deviceFFT_t* d_grid, float* d_tempgrid, int ng, double rl, int nbins, const char* fname, int nfolds, int blockSize, int calls = 0);

        void GetFinerPowerSpectrum(float4* d_temp_pos, int ng, double rl, int nbins, int fftNg, const char* fname, int blockSize);

        void GenerateDisplacementIC(const char* params_file, HACCGPM::serial::MemoryManager* mem, int ng, double rl, double z, double deltaT, double fscal, int seed, int blockSize, int calls = 0);

        void Solve(deviceFFT_t* d_rho, hostFFT_t* d_greens, int ng, int blockSize, int calls = 0);

        void SolveGradient(float4* d_grad, deviceFFT_t* d_rho, hostFFT_t* d_greens, int ng, int blockSize, int calls = 0);

        void InitGreens(hostFFT_t* d_greens, int ng, int blockSize, int calls = 0);

        void forward_fft(deviceFFT_t* data, deviceFFT_t* out, int ng, int calls = 0);

        void forward_fft(deviceFFT_t* data, int ng, int calls = 0);

        void backward_fft(deviceFFT_t* data, deviceFFT_t* out, int ng, int calls = 0);

        void backward_fft(deviceFFT_t* data, int ng, int calls = 0);

        void fft_cache_plan(int ng);

        void printCICTimes();

        void printFFTTimes();

        void printPowerTimes();

        void printOutputTimes();
    }
}