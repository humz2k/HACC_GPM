#include <stdlib.h>
#include <stdio.h>
#include "haccgpm.hpp"

double omega_nu_massive(double a, double m_omega_nu, double m_f_nu_massive, double m_omega_radiation){
    double mat = m_omega_nu/pow(a,3.0);
    double rad = m_f_nu_massive*m_omega_radiation/pow(a,4.0);
    return (mat>=rad)*mat + (rad>mat)*rad;
}

double HACCGPM::serial::get_adot(double m_aa, double m_w, double m_wa, double m_omega_cb, double m_omega_radiation, double m_f_nu_massless, double m_f_nu_massive, double m_omega_matter, double m_omega_nu){
    double pp1 = pow(m_aa, -3.0*(m_w+m_wa))*exp(-3.0*m_wa*(1.0-m_aa));
    double tmp = m_omega_cb
    + (1.0+m_f_nu_massless)*m_omega_radiation/m_aa
    + omega_nu_massive(m_aa,m_omega_nu,m_f_nu_massive,m_omega_radiation)*pow(m_aa,3.0)
    + (1.0-m_omega_matter-(1.0+m_f_nu_massless)*m_omega_radiation)*pp1;
    tmp /= m_aa;
    return sqrt(tmp);
}

double HACCGPM::serial::get_fscal(double m_aa, double m_adot, double m_omega_cb){
    float dtdy = m_aa/(m_aa*m_adot);
    double m_phiscal = 1.5*m_omega_cb;//Poisson equation is grad^2 phi = 3/2 omega_m (rho-1)
    double m_fscal = m_phiscal*dtdy*(1.0/m_aa);
    return m_fscal;
}


HACCGPM::Timestepper::Timestepper(HACCGPM::Params iparams){
    world_rank = iparams.world_rank;
    deltaT = (((1/(iparams.z_fin + 1)) - (1/(iparams.z_ini + 1))) / (double)iparams.nsteps);
    params = iparams;
}

void HACCGPM::Timestepper::setInitialA(double a){
    aa = a;
    z = (1.0f/aa) - 1;
    adot = HACCGPM::serial::get_adot(aa,params.m_w_de,params.m_wa_de,params.m_omega_cb,params.m_omega_radiation,params.m_f_nu_massless,params.m_f_nu_massive,params.m_omega_matter,params.m_omega_nu);
    fscal = HACCGPM::serial::get_fscal(aa,adot,params.m_omega_cb);
}

void HACCGPM::Timestepper::setInitialZ(double z){
    z = z;
    aa = 1.0/(z+1.0);
    adot = HACCGPM::serial::get_adot(aa,params.m_w_de,params.m_wa_de,params.m_omega_cb,params.m_omega_radiation,params.m_f_nu_massless,params.m_f_nu_massive,params.m_omega_matter,params.m_omega_nu);
    fscal = HACCGPM::serial::get_fscal(aa,adot,params.m_omega_cb);
    if(world_rank == 0)printf("Timestepper: a=%g, z=%g, adot=%g, fscal=%g\n",aa,z,adot,fscal);
}

void HACCGPM::Timestepper::advanceHalfStep(){
    aa += deltaT * 0.5;
    z = (1.0f/aa) - 1;
    adot = HACCGPM::serial::get_adot(aa,params.m_w_de,params.m_wa_de,params.m_omega_cb,params.m_omega_radiation,params.m_f_nu_massless,params.m_f_nu_massive,params.m_omega_matter,params.m_omega_nu);
    fscal = HACCGPM::serial::get_fscal(aa,adot,params.m_omega_cb);
    if(world_rank == 0)printf("Timestepper: a=%g, z=%g, adot=%g, fscal=%g\n",aa,z,adot,fscal);
}

void HACCGPM::Timestepper::reverseHalfStep(){
    aa -= deltaT * 0.5;
    z = (1.0f/aa) - 1;
    adot = HACCGPM::serial::get_adot(aa,params.m_w_de,params.m_wa_de,params.m_omega_cb,params.m_omega_radiation,params.m_f_nu_massless,params.m_f_nu_massive,params.m_omega_matter,params.m_omega_nu);
    fscal = HACCGPM::serial::get_fscal(aa,adot,params.m_omega_cb);
}