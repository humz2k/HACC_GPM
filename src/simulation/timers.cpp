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

void HACCGPM::parallel::printTimingStats(const char *preamble, // text at beginning of line
		      double dt)            // delta t in seconds
{
  int myrank, nranks;
  double max, min, sum, avg, var, stdev;

  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  MPI_Allreduce(&dt, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&dt, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&dt, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  avg = sum/nranks;

  dt -= avg;
  dt *= dt;
  MPI_Allreduce(&dt, &var, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  var *= 1.0/nranks;
  stdev = sqrt(var);

  if(myrank==0) {
    printf("%s  max %.3es  avg %.3es  min %.3es  dev %.3es\n",
	   preamble, max, avg, min, stdev);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  return;
}

void HACCGPM::serial::printTimers(CPUTimer_t init, CPUTimer_t total){
  printf("\n\n=========\nTimers:\n");
    HACCGPM::serial::printCICTimes();
    HACCGPM::serial::printFFTTimes();
    HACCGPM::serial::printPowerTimes();
    HACCGPM::serial::printOutputTimes();
    printf("   Initialization: %llu us (%5.2g minutes)\n",init,((double)(init)) * 1.66667e-8);
    printf("   Total: %5.2g minutes\n",((double)(total)) * 1.66667e-8);
    printf("=========\n\n");
}