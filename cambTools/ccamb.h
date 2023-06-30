extern "C" void get_pk(const char* params_file, double* grid, double z, int ng, double rl, int calls);

extern "C" void get_delta_and_dotDelta(const char* params_file, double z, double z1, double* delta, double* dotDelta, int calls);

extern "C" void init_python(int calls);

extern "C" void finalize_python(int calls);