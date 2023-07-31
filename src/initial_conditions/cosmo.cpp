#include <stdlib.h>
#include <stdio.h>
//#include <cstdlib>
#include <mpi.h>
#include "haccgpm.hpp"

//#define VerboseCosmo

HACCGPM::CosmoClass::CosmoClass(HACCGPM::Params& params){

    Omega_m = params.m_omega_matter;
    Omega_cdm = params.m_omega_cdm;
    Omega_bar = params.m_omega_baryon;
    Omega_cb = params.m_omega_cb;
    Omega_nu = params.m_omega_nu;
    f_nu_massless = params.m_f_nu_massless;
    f_nu_massive = params.m_f_nu_massive;
    Omega_r = params.m_omega_radiation;
    h = params.m_hubble;
    w_de = params.m_w_de;
    wa_de = params.m_wa_de;
    strcpy(ipk,params.ipk);
    
}

void HACCGPM::CosmoClass::read_ipk(double** out, int* nbins, double* k_delta, double* k_max, double* k_min, int calls){
    

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    #ifdef VerboseCosmo
    getIndent(calls);
    #else
    char indent[] = "";
    #endif

    if(world_rank == 0)printf("%sReading ipk\n",indent);
    if(world_rank == 0)printf("%s   fname: %s\n",indent,ipk);
    double header[20];
    FILE *ptr;
    ptr = fopen(ipk,"rb");
    if (ptr == NULL){
      printf("Can't open %s\n",ipk);
      exit(1);
    }
    fread(header,sizeof(header),1,ptr);
    if(world_rank == 0){
      printf("%s   k_min          = %g\n",indent,header[0]);
      printf("%s   k_max          = %g\n",indent,header[1]);
      printf("%s   k_bins         = %g\n",indent,header[2]);
      printf("%s   k_delta        = %g\n",indent,header[3]);
      printf("%s   OMEGA_CDM      = %g\n",indent,header[4]);
      printf("%s   DEUT           = %g\n",indent,header[5]);
      printf("%s   OMEGA_NU       = %g\n",indent,header[6]);
      printf("%s   HUBBLE         = %g\n",indent,header[7]);
      printf("%s   SS8            = %g\n",indent,header[8]);
      printf("%s   NS             = %g\n",indent,header[9]);
      printf("%s   W_DE           = %g\n",indent,header[10]);
      printf("%s   WA_DE          = %g\n",indent,header[11]);
      printf("%s   T_CMB          = %g\n",indent,header[12]);
      printf("%s   N_EFF_MASSLESS = %g\n",indent,header[13]);
      printf("%s   N_EFF_MASSIVE  = %g\n",indent,header[14]);
      printf("%s   Z_IN           = %g\n",indent,header[15]);
    }
    int _k_bins = header[2];
    double _k_delta = header[3];
    double _k_max = header[1];
    double _k_min = header[0];
    double* tmp = (double*)malloc(sizeof(double)*(_k_bins+20));
    fread(tmp,sizeof(double),_k_bins,ptr);
    *out = tmp;
    *nbins = _k_bins;
    *k_delta = _k_delta;
    *k_max = _k_max;
    *k_min = _k_min;
    fclose(ptr);
    

}

float HACCGPM::CosmoClass::da_dtau(float a, float OmM, float OmL){
    float da_dtau_2 = 1+OmM*((1/a)-1) + OmL*((a*a)-1);
    return sqrt(da_dtau_2);
}

float HACCGPM::CosmoClass::da_dtau__3(float a, float OmM, float OmL){
    float da_dtau_1 = HACCGPM::CosmoClass::da_dtau(a,OmM,OmL);
    return 1/(da_dtau_1*da_dtau_1*da_dtau_1);
}

float HACCGPM::CosmoClass::int_1_da_dtau_3(float a, float OmM, float OmL, int bins){
    float start = 0;
    float end = a;
    float delta = (end-start)/((float)bins);
    float sum = 0;
    for (int k = 1; k < bins; k++){
        sum += da_dtau__3(start + ((float)k)*delta,OmM,OmL);
    }
    sum += (da_dtau__3(start,OmM,OmL) + da_dtau__3(end,OmM,OmL))/2.0f;
    sum *= delta;
    return sum;
    //float da_dtau_1 = HACCGPM::CosmoClass::da_dtau(a,OmM,OmL);
    //return 1/(sqrt(da_dtau_1*da_dtau_1*da_dtau_1));
}

float HACCGPM::CosmoClass::delta(float a, float OmM, float OmL){
    float integral = int_1_da_dtau_3(a,OmM,OmL,100);
    float diff = da_dtau(a,OmM,OmL);
    float mul = (5*OmM)/(2*a);
    return mul*diff*integral;
}

float HACCGPM::CosmoClass::dotDelta(float a, float OmM, float OmL, float h){
    return (delta(a+h,OmM,OmL)-delta(a-h,OmM,OmL))/(2.0f*h);
}

float HACCGPM::CosmoClass::z2a(float z){
    return 1.0f/(1.0f+z);
}

float HACCGPM::CosmoClass::a2z(float a){
    return (1.0f/(a)) - 1.0f;
}

void HACCGPM::CosmoClass::get_delta_and_dotDelta(float z, float z1, float* d, float* d_dot){
    float OmM = Omega_m;
    float OmL = 1-(OmM + Omega_nu);
    //printf("OmM = %f, OmL = %f\n",OmM,OmL);
    *d = delta(z2a(z),OmM,OmL);
    *d_dot = dotDelta(z2a(z1),OmM,OmL,0.0001);
}

void HACCGPM::CosmoClass::get_delta_and_dotDelta(float z, float z1, double* d, double* d_dot){
    float OmM = Omega_m;
    float OmL = 1-(OmM + Omega_nu);
    //printf("OmM = %f, OmL = %f\n",OmM,OmL);
    *d = delta(z2a(z),OmM,OmL);
    *d_dot = dotDelta(z2a(z1),OmM,OmL,0.0001);
}

void HACCGPM::CosmoClass::rkck(float* y, float* dydx, int n, float x, float h,
		      float* yout, float* yerr,
		      void (HACCGPM::CosmoClass::*derivs)(float, float*, float*))
{
  int i;
  static float a2=0.2,a3=0.3,a4=0.6,a5=1.0,a6=0.875,b21=0.2,
    b31=3.0/40.0,b32=9.0/40.0,b41=0.3,b42 = -0.9,b43=1.2,
    b51 = -11.0/54.0, b52=2.5,b53 = -70.0/27.0,b54=35.0/27.0,
    b61=1631.0/55296.0,b62=175.0/512.0,b63=575.0/13824.0,
    b64=44275.0/110592.0,b65=253.0/4096.0,c1=37.0/378.0,
    c3=250.0/621.0,c4=125.0/594.0,c6=512.0/1771.0,
    dc5 = -277.00/14336.0;
  float dc1=c1-2825.0/27648.0,dc3=c3-18575.0/48384.0,
    dc4=c4-13525.0/55296.0,dc6=c6-0.25;
  float *ak2,*ak3,*ak4,*ak5,*ak6,*ytemp;

  ak2= (float *)malloc(n*sizeof(float));
  ak3= (float *)malloc(n*sizeof(float));
  ak4= (float *)malloc(n*sizeof(float));
  ak5= (float *)malloc(n*sizeof(float));
  ak6= (float *)malloc(n*sizeof(float));
  ytemp= (float *)malloc(n*sizeof(float));

  for (i=0; i<n; ++i)
    ytemp[i]=y[i]+b21*h*dydx[i];
  (this->*derivs)(x+a2*h,ytemp,ak2);
  for (i=0; i<n; ++i)
    ytemp[i]=y[i]+h*(b31*dydx[i]+b32*ak2[i]);
  (this->*derivs)(x+a3*h,ytemp,ak3);
  for (i=0; i<n; ++i)
    ytemp[i]=y[i]+h*(b41*dydx[i]+b42*ak2[i]+b43*ak3[i]);
  (this->*derivs)(x+a4*h,ytemp,ak4);
  for (i=0; i<n; ++i)
    ytemp[i]=y[i]+h*(b51*dydx[i]+b52*ak2[i]+b53*ak3[i]+b54*ak4[i]);
  (this->*derivs)(x+a5*h,ytemp,ak5);
  for (i=0; i<n; ++i)
    ytemp[i]=y[i]+h*(b61*dydx[i]+b62*ak2[i]+b63*ak3[i]+b64*ak4[i]+b65*ak5[i]);
  (this->*derivs)(x+a6*h,ytemp,ak6);
  for (i=0; i<n; ++i)
    yout[i]=y[i]+h*(c1*dydx[i]+c3*ak3[i]+c4*ak4[i]+c6*ak6[i]);
  for (i=0; i<n; ++i)
    yerr[i]=h*(dc1*dydx[i]+dc3*ak3[i]+dc4*ak4[i]+dc5*ak5[i]+dc6*ak6[i]);

  free(ytemp);
  free(ak6);
  free(ak5);
  free(ak4);
  free(ak3);
  free(ak2);
}

#define SAFETY 0.9
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
static float maxarg1,maxarg2, minarg1, minarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ? (maxarg1) : (maxarg2))
#define FMIN(a,b) (minarg1=(a),minarg2=(b),(minarg1) < (minarg2) ? (minarg1) : (minarg2))
void HACCGPM::CosmoClass::rkqs(float* y, float* dydx, int n, float *x, float htry,
		      float eps,
                      float* yscal, float *hdid, float *hnext, int *feval,
                      void (HACCGPM::CosmoClass::*derivs)(float, float *, float *))
{
  int i;
  float errmax,h,htemp,xnew,*yerr,*ytemp;

  yerr= (float *)malloc(n*sizeof(float));
  ytemp= (float *)malloc(n*sizeof(float));
  h=htry;

  for (;;) {
    rkck(y,dydx,n,*x,h,ytemp,yerr,derivs);
    *feval += 5;
    errmax=0.0;
    for (i=0; i<n; ++i) {errmax=FMAX(errmax,fabs(yerr[i]/yscal[i]));}
    errmax /= eps;
    if (errmax <= 1.0) break;
    htemp=SAFETY*h*pow((double) errmax,PSHRNK);
    h=(h >= 0.0 ? FMAX(htemp,0.1*h) : FMIN(htemp,0.1*h));
    xnew=(*x)+h;
    if (xnew == *x) {
      printf("Stepsize underflow in ODEsolve rkqs");
      exit(1);
    }
  }
  if (errmax > ERRCON) *hnext=SAFETY*h*pow((double) errmax,PGROW);
  else *hnext=5.0*h;
  *x += (*hdid=h);
  for (i=0; i<n; ++i) {y[i]=ytemp[i];}
  free(ytemp);
  free(yerr);
}
#undef SAFETY
#undef PGROW
#undef PSHRNK
#undef ERRCON
#undef FMAX
#undef FMIN

#define MAXSTP 10000
#define TINY 1.0e-30
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
void HACCGPM::CosmoClass::odesolve(float* ystart, int nvar, float x1, float x2,
			  float eps, float h1,
                          void (HACCGPM::CosmoClass::*derivs)(float, float *, float *),
			  bool print_stat)
{
  int i, nstp, nok, nbad, feval;
  float x,hnext,hdid,h;
  float *yscal,*y,*dydx;
  const float hmin=0.0;

  feval = 0;
  yscal= (float *)malloc(nvar*sizeof(float));
  y= (float *)malloc(nvar*sizeof(float));
  dydx= (float *)malloc(nvar*sizeof(float));
  x=x1;
  h=SIGN(h1,x2-x1);
  nok = nbad = 0;
  for (i=0; i<nvar; ++i) {y[i]=ystart[i];}

  for (nstp=0; nstp<MAXSTP; ++nstp) {
    (this->*derivs)(x, y, dydx);
    ++feval;
    for (i=0; i<nvar; ++i)
    {yscal[i]=fabs(y[i])+fabs(dydx[i]*h)+TINY;}
    if ((x+h-x2)*(x+h-x1) > 0.0) h=x2-x;
    rkqs(y,dydx,nvar,&x,h,eps,yscal,&hdid,&hnext,&feval,derivs);
    if (hdid == h) ++nok; else ++nbad;
    if ((x-x2)*(x2-x1) >= 0.0) {
      for (i=0; i<nvar; ++i) {ystart[i]=y[i];}
      free(dydx);
      free(y);
      free(yscal);
      if (print_stat){
	printf("ODEsolve:\n");
	printf(" Evolved from x = %f to x = %f\n", x1, x2);
	printf(" successful steps: %d\n", nok);
	printf(" bad steps: %d\n", nbad);
	printf(" function evaluations: %d\n", feval);
      }
      return;
    }
    if (fabs(hnext) <= hmin) {
      printf("Step size too small in ODEsolve");
      exit(1);
    }
    h=hnext;
  }
  printf("Too many steps in ODEsolve");
  exit(1);
}
#undef MAXSTP
#undef TINY
#undef SIGN

void HACCGPM::CosmoClass::GrowthFactor(float z, float *gf, float *g_dot) {
  float x1, x2, dplus, ddot;
  const float zinfinity = 100000.0;

  x1 = 1.0/(1.0+zinfinity);
  x2 = 1.0/(1.0+z);
  float ystart[2];
  ystart[0] = x1;
  ystart[1] = 0.0;

  odesolve(ystart, 2, x1, x2, 1.0e-6, 1.0e-6, &CosmoClass::growths, false);
  //printf("Dplus = %f;  Ddot = %f \n", ystart[0], ystart[1]);

  dplus = ystart[0];
  ddot  = ystart[1];
  x1 = 1.0/(1.0+zinfinity);
  x2 = 1.0;
  ystart[0] = x1;
  ystart[1] = 0.0;

  odesolve(ystart, 2, x1, x2, 1.0e-6, 1.0e-6, &CosmoClass::growths, false);
  //printf("Dplus = %f;  Ddot = %f \n", ystart[0], ystart[1]);

  *gf    = dplus/ystart[0];
  *g_dot = ddot/ystart[0];
  //printf("\nGrowth factor = %f;  Derivative = %f \n", dplus/ystart[0], ddot/ystart[0]);
}

void HACCGPM::CosmoClass::growths(float a, float* y, float* dydx) {
  float H;
  H = sqrt(Omega_cb/pow(a,3.0f)
	   + (1.0 + f_nu_massless)*Omega_r/pow(a,4.0f)
	   + Omega_nu_massive(a)
	   + (1.0 - Omega_m - (1.0+f_nu_massless)*Omega_r)
	   *pow(a,(-3.0*(1.0+w_de+wa_de)))*exp(-3.0*wa_de*(1.0-a))
      );
  dydx[0] = y[1]/(a*H);
  dydx[1] = -2.0*y[1]/a + 1.5*Omega_cb*y[0]/(H*pow(a, 4.0f));
}

float HACCGPM::CosmoClass::Omega_nu_massive(float a) {
  float mat = Omega_nu/pow(a,3.0f);
  float rad = f_nu_massive*Omega_r/pow(a,4.0f);
  return (mat>=rad)*mat + (rad>mat)*rad;
}