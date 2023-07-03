#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
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

            Timestepper(Params iparams);

            void setInitialA(double a);

            void setInitialZ(double z);

            void advanceHalfStep();

            void reverseHalfStep();
    };

    namespace parallel{
        class MemoryManager{
            float4* d_pos;
            float4* d_vel;
            float4* d_grad;
            hostFFT_t* d_greens;
            deviceFFT_t* d_grid1;
            deviceFFT_t* d_grid2;

            MemoryManager(HACCGPM::Params params);

            ~MemoryManager();
        };
    }

    namespace serial{

        class MemoryManager{
            public:
                float4* d_pos;
                float4* d_vel;
                float4* d_grad;
                hostFFT_t* d_greens;
                deviceFFT_t* d_grid;

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

        double get_fscal(double m_aa, double m_adot, double m_omega_cb);

        double get_adot(double m_aa, double m_w, double m_wa, double m_omega_cb, double m_omega_radiation, double m_f_nu_massless, double m_f_nu_massive, double m_omega_matter, double m_omega_nu);

        void writeOutput(char* fname, float4* d_pos, float4* d_vel, int ng, int calls = 0);

        void GetPowerSpectrum(float4* d_pos, deviceFFT_t* d_grid, int ng, double rl, int nbins, const char* fname, int nfolds, int blockSize, int calls = 0);

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