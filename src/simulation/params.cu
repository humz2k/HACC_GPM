#include <stdlib.h>
#include <stdio.h>
#include "haccgpm.hpp"

HACCGPM::Params HACCGPM::read_params(const char* fname){
    FILE *fp;
    fp = fopen(fname, "r");

    if(fp == NULL){printf("file %s not found", fname);}

    char parameter[200];
    char value[200];

    char tempbuff[400];

    HACCGPM::Params out;
    strcpy(out.fname,fname);
    out.do_analysis = false;
    out.frac = 1.5;
    out.world_rank = 0;
    out.dump_init = false;
    out.dump_final = false;
    out.ol = 1;

    for (int i = 0; i < MAX_STEPS; i++){
        out.pks[i] = false;
        out.dumps[i] = false;
        out.analysis[i] = false;
    }

    out.lastStep = -1;

    while(!feof(fp)) 
    {
        if (fgets(tempbuff,400,fp)) {
            if (strlen(tempbuff) != 1){
                sscanf(tempbuff, "%199s %199s", parameter, value);
                //printf("%s %s\n",parameter,value);
                if (strcmp(parameter,"OMEGA_CDM") == 0){
                    float val = atof(value);
                    out.m_omega_cdm = val;
                } else if (strcmp(parameter,"DEUT") == 0){
                    float val = atof(value);
                    out.m_deut = val;
                } else if (strcmp(parameter,"OMEGA_NU") == 0){
                    float val = atof(value);
                    out.m_omega_nu = val;
                } else if (strcmp(parameter,"HUBBLE") == 0){
                    float val = atof(value);
                    out.m_hubble = val;
                } else if (strcmp(parameter,"SS8") == 0){
                    float val = atof(value);
                    out.m_ss8 = val;
                } else if (strcmp(parameter,"NS") == 0){
                    float val = atof(value);
                    out.m_ns = val;
                } else if (strcmp(parameter,"W_DE") == 0){
                    float val = atof(value);
                    out.m_w_de = val;
                } else if (strcmp(parameter,"WA_DE") == 0){
                    float val = atof(value);
                    out.m_wa_de = val;
                } else if (strcmp(parameter,"T_CMB") == 0){
                    float val = atof(value);
                    out.m_Tcmb = val;
                } else if (strcmp(parameter,"N_EFF_MASSLESS") == 0){
                    float val = atof(value);
                    out.m_neff_massless = val;
                } else if (strcmp(parameter,"N_EFF_MASSIVE") == 0){
                    float val = atof(value);
                    out.m_neff_massive = val;
                } else if (strcmp(parameter,"OUTPUT_BASE_NAME") == 0){
                    strcpy(out.prefix,value);
                } else if (strcmp(parameter,"Z_IN") == 0){
                    float val = atof(value);
                    out.z_ini = val;
                } else if (strcmp(parameter,"Z_FIN") == 0){
                    float val = atof(value);
                    out.z_fin = val;
                } else if (strcmp(parameter,"N_STEPS") == 0){
                    int val = atoi(value);
                    out.nsteps = val;
                } else if (strcmp(parameter,"NG") == 0){
                    int val = atoi(value);
                    out.ng = val;
                } else if (strcmp(parameter,"RL") == 0){
                    float val = atof(value);
                    out.rl = val;
                } else if (strcmp(parameter,"SEED") == 0){
                    int val = atoi(value);
                    out.seed = val;
                } else if (strcmp(parameter,"BLOCK_SIZE") == 0){
                    int val = atoi(value);
                    out.blockSize = val;
                } else if (strcmp(parameter,"LAST_STEP") == 0){
                    int val = atoi(value);
                    out.lastStep = val;
                } else if (strcmp(parameter,"PK_FOLDS") == 0){
                    int val = atoi(value);
                    out.pkFolds = val;
                } else if (strcmp(parameter,"PK_DUMP") == 0){
                    //printf("%lu\n",strlen(value));
                    //printf("%s\n",value);
                    char* token = strtok(value, ",");
                    while (token != NULL) {
                        int val = atoi(token);
                        //printf("%d VAL\n",val);
                        out.pks[val] = true;
                        token = strtok(NULL, ",");
                    }
                } else if (strcmp(parameter,"PARTICLE_DUMP") == 0){
                    char* token = strtok(value, ",");
                    while (token != NULL) {
                        int val = atoi(token);
                        out.dumps[val] = true;
                        token = strtok(NULL, ",");
                    }
                } else if (strcmp(parameter,"ANALYSIS_STEPS") == 0){
                    //printf("ANALYSIS: %s\n",value);
                    if (strcmp(value,"all") == 0){
                        for (int i = 0; i < MAX_STEPS; i++){
                            out.analysis[i] = true;
                        }
                    } else{
                        char* token = strtok(value, ",");
                        while (token != NULL) {
                            int val = atoi(token);
                            out.analysis[val] = true;
                            token = strtok(NULL, ",");
                        }
                    }
                } else if (strcmp(parameter,"ANALYSIS_DIR") == 0){
                    strcpy(out.analysis_dir,value);
                } else if (strcmp(parameter,"ANALYSIS_PY") == 0){
                    strcpy(out.analysis_py,value);
                } else if (strcmp(parameter,"DO_ANALYSIS") == 0){
                    out.do_analysis = true;
                } else if (strcmp(parameter,"DUMP_INIT") == 0){
                    out.dump_init = true;
                } else if (strcmp(parameter,"DUMP_FINAL") == 0){
                    out.dump_final = true;
                } else if (strcmp(parameter,"IPK") == 0){
                    strcpy(out.ipk,value);
                } else if (strcmp(parameter,"OL") == 0){
                    out.ol = atof(value);
                }
            }
        }
    }

    if (out.lastStep == -1){
        out.lastStep = out.nsteps;
    }

    fclose(fp);

    out.overload = ceil((((double)out.ng) / (out.rl)) * out.ol);

    out.m_omega_baryon = out.m_deut / out.m_hubble / out.m_hubble;
    out.m_omega_cb = out.m_omega_cdm + out.m_omega_baryon;
    out.m_omega_matter = out.m_omega_cb + out.m_omega_nu;

    out.m_omega_radiation = 2.471e-5*pow(out.m_Tcmb/2.725f,4.0f)/pow(out.m_hubble,2.0f);
    out.m_f_nu_massless = out.m_neff_massless*7.0/8.0*pow(4.0/11.0,4.0/3.0);
    out.m_f_nu_massive = out.m_neff_massive*7.0/8.0*pow(4.0/11.0,4.0/3.0);

    return out;
}

void print_params(HACCGPM::Params params){
    printf("OMEGA_CDM %g\n",params.m_omega_cdm);
    printf("DEUT %g\n",params.m_deut);
    printf("OMEGA_NU %g\n",params.m_omega_nu);
    printf("HUBBLE %g\n",params.m_hubble);
    printf("SS8 %g\n",params.m_ss8);
    printf("NS %g\n",params.m_ns);
    printf("W_DE %g\n",params.m_w_de);
    printf("WA_DE %g\n",params.m_wa_de);
    printf("T_CMB %g\n",params.m_Tcmb);
    printf("N_EFF_MASSLESS %g\n",params.m_neff_massless);
    printf("N_EFF_MASSIVE %g\n",params.m_neff_massive);
    printf("OMEGA_BARYON %g\n",params.m_omega_baryon);
    printf("OMEGA_CB %g\n",params.m_omega_cb);
    printf("OMEGA_MATTER %g\n",params.m_omega_matter);
    printf("OMEGA_RADIATION %g\n",params.m_omega_radiation);
    printf("M_F_NU_MASSLESS %g\n",params.m_f_nu_massless);
    printf("M_F_NU_MASSIVE %g\n",params.m_f_nu_massive);
    printf("OUTPUT_BASE_NAME %s\n",params.prefix);
    printf("Z_IN %g\n",params.z_ini);
    printf("Z_FIN %g\n",params.z_fin);
    printf("N_STEPS %d\n",params.nsteps);
    printf("NG %d\n",params.ng);
    printf("RL %g\n",params.rl);
    printf("SEED %d\n",params.seed);
    printf("BLOCK_SIZE %d\n",params.blockSize);
}