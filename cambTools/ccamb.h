extern "C" void get_pk(const char* params_file, double* grid, double z, int ng, double rl, int calls);

extern "C" void get_delta_and_dotDelta(const char* params_file, double z, double z1, double* delta, double* dotDelta, int calls);

extern "C" void init_python(int calls, int world_rank_);

extern "C" void finalize_python(int calls);

extern "C" void get_pk_parallel(const char* params_file, double* grid, double z, int ng, double rl, int nlocal, int world_rank, int calls);