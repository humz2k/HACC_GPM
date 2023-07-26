#include <stdlib.h>
#include <stdio.h>
#include "ic_kernels.hpp"
#include <math.h>

//#define NOPYTHON

#define VerboseInitializer

void GenerateFourierAmplitudes(HACCGPM::CosmoClass& cosmo, HACCGPM::Params& params, deviceFFT_t* d_grid1, hostFFT_t* d_pkScale, double z, int calls){
    int numBlocks = (params.ng*params.ng*params.ng)/params.blockSize;

    getIndent(calls);

    #ifdef VerboseInitializer
    printf("%sGenerateFourierAmplitudes was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   z %g\n",indent,indent,params.blockSize,indent,numBlocks,indent,z);
    printf("%s   Calling generate_rng...\n",indent);
    #endif

    launch_generate_rng(d_grid1,params.ng,params.seed,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    printf("%s      Called generate_rng.\n",indent);
    printf("%s   Doing Forward FFT...\n",indent);
    #endif

    HACCGPM::serial::forward_fft(d_grid1,params.ng,calls+1);

    #ifdef VerboseInitializer
    printf("%s      Done Forward FFT...\n",indent);
    
    #endif

    #ifdef NOPYTHON
        #ifdef VerboseInitializer
        printf("%s   Getting Pk from ipk...\n",indent);
        #endif

        launch_interpolate_pk(cosmo,d_pkScale,params.ng,params.rl,numBlocks,params.blockSize,calls);

    #else
        #ifdef VerboseInitializer
        printf("%s   Getting Pk from Camb...\n",indent);
        #endif

        launch_get_pk(d_pkScale,z,params.fname,params.ng,params.rl,calls);

        #ifdef VerboseInitializer
        printf("%s      Got Pk from Camb.\n",indent);
        #endif
    #endif
    
    #ifdef VerboseInitializer
    printf("%s   Scaling Amplitudes...\n",indent);
    #endif

    launch_scale_amplitudes(d_grid1,d_pkScale,params.ng*params.ng*params.ng,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    printf("%s      Scaled Amplitudes.\n",indent);
    #endif
}

void HACCGPM::serial::GenerateDisplacementIC(HACCGPM::serial::MemoryManager& mem, HACCGPM::CosmoClass& cosmo, HACCGPM::Params& params, HACCGPM::Timestepper& ts, int calls){
    int numBlocks = (params.ng*params.ng*params.ng)/params.blockSize;
    getIndent(calls);

    #ifdef VerboseInitializer
    printf("%sGenerateDisplacementIC was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   params %s\n",indent,indent,params.blockSize,indent,numBlocks,indent,params.fname);
    printf("%s   Calling GenerateFourierAmplitudes...\n",indent);
    #endif

    GenerateFourierAmplitudes(cosmo, params, mem.d_grid, mem.d_greens, params.z_ini, calls+1);

    #ifdef VerboseInitializer
    printf("%s      Called GenerateFourierAmplitudes.\n",indent);
    printf("%s   Calling get_delta_and_dotDelta...\n",indent);
    #endif

    double delta;
    double dotDelta;
    double this_a = (1/(params.z_ini + 1)) - (ts.deltaT/2.0f);
    double velZ = (1.0f/this_a) - 1.0f;

    cosmo.get_delta_and_dotDelta(params.z_ini,velZ,&delta,&dotDelta);
    
    printf("%s      Delta %g, dotDelta %g\n",indent,delta,dotDelta);

    #ifdef VerboseInitializer
    printf("%s      Called get_delta_and_dotDelta.\n",indent);
    printf("%s   Calling transformDensityField...\n",indent);
    #endif

    launch_transform_density_field(mem.d_grid,mem.d_x,mem.d_y,mem.d_z,delta,params.rl,params.z_ini,params.ng,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    printf("%s      Called transformVelocityField.\n",indent);
    printf("%s   Doing Backward FFTs...\n",indent);
    #endif

    HACCGPM::serial::backward_fft(mem.d_x,params.ng,calls+1);
    HACCGPM::serial::backward_fft(mem.d_y,params.ng,calls+1);
    HACCGPM::serial::backward_fft(mem.d_z,params.ng,calls+1);

    double scale_by = 1.0f/((double)(params.ng*params.ng*params.ng));
    int scale_n = params.ng*params.ng*params.ng;
    launch_scale_fft(mem.d_x,scale_by,scale_n,numBlocks,params.blockSize,calls);
    launch_scale_fft(mem.d_y,scale_by,scale_n,numBlocks,params.blockSize,calls);
    launch_scale_fft(mem.d_z,scale_by,scale_n,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    printf("%s      Done Backward FFTs.\n",indent);
    printf("%s   Calling placeParticles...\n",indent);
    #endif

    launch_place_particles(mem.d_pos,mem.d_vel,mem.d_x,mem.d_y,mem.d_z,delta,dotDelta,params.rl,params.z_ini,ts.deltaT,ts.fscal,params.ng,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    printf("%s      Called placeParticles.\n",indent);
    #endif
}


void GenerateFourierAmplitudesParallel(HACCGPM::CosmoClass& cosmo, HACCGPM::Params& params, deviceFFT_t* d_grid, hostFFT_t* d_pkScale, double z, int calls){
    int numBlocks = (params.nlocal + (params.blockSize - 1))/params.blockSize;

    int world_rank = params.world_rank;

    getIndent(calls);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%sGenerateFourierAmplitudesParallel was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   z %g\n",indent,indent,params.blockSize,indent,numBlocks,indent,z);
    if(world_rank == 0)printf("%s   Calling generate_rng...\n",indent);
    #endif

    launch_generate_rng(d_grid,params.ng,params.seed,params.nlocal,params.local_grid_size_vec,params.grid_coords_vec,params.world_rank,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Called generate_rng.\n",indent);
    if(world_rank == 0)printf("%s   Doing Forward FFT...\n",indent);
    #endif

    HACCGPM::parallel::forward_fft(d_grid,params.ng,calls+1);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Done Forward FFT...\n",indent);
    #endif
    #ifdef NOPYTHON
    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s   Getting Pk from ipk...\n",indent);
    #endif

    launch_interpolate_pk(cosmo,d_pkScale,params.ng,params.rl,params.nlocal,params.local_grid_size_vec,params.grid_coords_vec,params.grid_dims_vec,params.world_rank,numBlocks,params.blockSize,calls);
    
    #else
    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s   Getting Pk from Camb...\n",indent);
    #endif
    launch_get_pk(d_pkScale,z,params.fname,params.ng,params.rl,params.nlocal,params.world_rank,calls);
    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Got Pk from Camb.\n",indent);
    #endif
    #endif

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s   Scaling Amplitudes...\n",indent);
    #endif

    launch_scale_amplitudes(d_grid,d_pkScale,params.nlocal,world_rank,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Scaled Amplitudes.\n",indent);
    #endif
}

void HACCGPM::parallel::GenerateDisplacementIC(HACCGPM::parallel::MemoryManager& mem, HACCGPM::CosmoClass& cosmo, HACCGPM::Params& params, HACCGPM::Timestepper& ts, int calls){

    int world_rank = params.world_rank;

    int numBlocks = (params.nlocal)/params.blockSize;
    getIndent(calls);

    #ifdef VerboseInitializer
    if (params.world_rank == 0)printf("%sGenerateDisplacementIC was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   params %s\n",indent,indent,params.blockSize,indent,numBlocks,indent,params.fname);
    if (params.world_rank == 0)printf("%s   Calling GenerateFourierAmplitudesParallel...\n",indent);
    #endif

    GenerateFourierAmplitudesParallel(cosmo, params, mem.d_grid, mem.d_greens, params.z_ini, calls+1);

    #ifdef VerboseInitializer
    if (params.world_rank == 0)printf("%s      Called GenerateFourierAmplitudes.\n",indent);
    if (params.world_rank == 0)printf("%s   Calling get_delta_and_dotDelta...\n",indent);
    #endif

    double delta;
    double dotDelta;
    double this_a = (1/(params.z_ini + 1)) - (ts.deltaT/2.0f);
    double velZ = (1.0f/this_a) - 1.0f;
    cosmo.get_delta_and_dotDelta(params.z_ini,velZ,&delta,&dotDelta);

    if (params.world_rank == 0)printf("%s      Delta %g, dotDelta %g\n",indent,delta,dotDelta);

    #ifdef VerboseInitializer
    if (params.world_rank == 0)printf("%s      Called get_delta_and_dotDelta.\n",indent);
    if (params.world_rank == 0)printf("%s   Calling transformDensityField...\n",indent);
    #endif

    launch_transform_density_field(mem.d_grid,mem.d_x,mem.d_y,mem.d_z,delta,params.rl,params.z_ini,params.ng,params.nlocal,params.local_grid_size_vec,params.grid_coords_vec,params.grid_dims_vec,params.world_rank,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    if (params.world_rank == 0)printf("%s      Called transformVelocityField.\n",indent);
    if (params.world_rank == 0)printf("%s   Doing Backward FFTs...\n",indent);
    #endif

    HACCGPM::parallel::backward_fft(mem.d_x,params.ng,calls+1);
    HACCGPM::parallel::backward_fft(mem.d_y,params.ng,calls+1);
    HACCGPM::parallel::backward_fft(mem.d_z,params.ng,calls+1);

    double scale_by = 1.0f/((double)(params.ng*params.ng*params.ng));

    launch_scale_fft(mem.d_x,scale_by,params.nlocal,params.world_rank,numBlocks,params.blockSize,calls);
    launch_scale_fft(mem.d_y,scale_by,params.nlocal,params.world_rank,numBlocks,params.blockSize,calls);
    launch_scale_fft(mem.d_z,scale_by,params.nlocal,params.world_rank,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    if (params.world_rank == 0)printf("%s      Done Backward FFTs.\n",indent);
    if (params.world_rank == 0)printf("%s   Calling placeParticles.\n",indent);
    #endif

    launch_place_particles(mem.d_pos,mem.d_vel,mem.d_x,mem.d_y,mem.d_z,delta,dotDelta,params.rl,params.z_ini,ts.deltaT,ts.fscal,params.ng,params.nlocal,params.local_grid_size_vec,params.world_rank,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    if (params.world_rank == 0)printf("%s      Called placeParticles.\n",indent);
    #endif

}