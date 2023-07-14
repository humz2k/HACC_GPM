#include <stdio.h>
#include <stdlib.h>
#include "haccgpm.hpp"
#include <mpi.h>

void HACCGPM::parallel::timing_stats(CPUTimer_t t_, CPUTimer_t* mint, CPUTimer_t* maxt, CPUTimer_t* meant){
    MPI_Barrier(MPI_COMM_WORLD);
    CPUTimer_t t = t_;
    MPI_Reduce(&t,mint,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    t = t_;
    MPI_Reduce(&t,maxt,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    t = t_;
    MPI_Reduce(&t,meant,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
    *meant /= comm_size;
}