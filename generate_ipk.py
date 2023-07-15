#!/home/hqureshi/miniconda3/bin/python

import camb
import numpy as np
from camb import model
import scipy.integrate
import scipy.misc
import warnings
import sys

warnings.filterwarnings('ignore')

def read_params(fname):
    with open(fname,'r') as f:
        raw = f.read().splitlines()

    data = {}
    for i in raw:
        if (len(i.strip()) == 0) or (i.strip().startswith('#')):
            pass
        else:
            try:
                name,val = i.split()
                data[name] = float(val)
            except:
                try:
                    name,val = i.split()
                    data[name] = val
                except:
                    pass
    
    return data

def hacc2camb(params):
    h = params['HUBBLE']
    Om_c = params['OMEGA_CDM']
    Om_b = params['DEUT'] / (h**2)
    Om_nu = params['OMEGA_NU']
    n_eff_massless = params['N_EFF_MASSLESS']
    n_eff_massive = params['N_EFF_MASSIVE']
    w_0 = params['W_DE']
    w_a = params['WA_DE']
    T_cmb = params['T_CMB']
    n_s = params['NS']
    
    omch2 = Om_c * (h**2)
    ombh2 = Om_b * (h**2)
    omnuh2 = Om_nu * (h**2)
    h = h * 100
    return {'H0': h,'ombh2': ombh2,'omch2':omch2}, n_s

def hacc2cosmology(params):
    h = params['HUBBLE']
    Om_c = params['OMEGA_CDM']
    Om_b = params['DEUT'] / (h**2)
    Om_nu = params['OMEGA_NU']
    n_eff_massless = params['N_EFF_MASSLESS']
    n_eff_massive = params['N_EFF_MASSIVE']
    w_0 = params['W_DE']
    w_a = params['WA_DE']
    T_cmb = params['T_CMB']
    n_s = params['NS']
    
    omch2 = Om_c * (h**2)
    ombh2 = Om_b * (h**2)
    omnuh2 = Om_nu * (h**2)
    h = h * 100
    return omch2,ombh2,omnuh2

def initcambpy():
    fname = 'params'
    #global PK
    #print('Initializing Camb')
    #haccparams = read_params(fname)
    #cambparams,ns = hacc2camb(haccparams)
    #pars = camb.CAMBparams()
    #pars.set_cosmology(**cambparams)
    #pars.InitPower.set_params(ns=ns)
    #pars.set_matter_power(redshifts=[200], kmax=10.0)
    #results = camb.get_results(pars)
    #pars.NonLinear = model.NonLinear_none
    #PK = results.get_matter_power_interpolator()
    return None

def get_pk(z,k,fname):
    #print('FNAME:',fname)
    haccparams = read_params(fname)
    cambparams,ns = hacc2camb(haccparams)
    pars = camb.CAMBparams()
    pars.set_cosmology(**cambparams)
    pars.InitPower.set_params(ns=ns)
    pars.set_matter_power(redshifts=[z], kmax=np.max(k))
    results = camb.get_results(pars)
    pars.NonLinear = model.NonLinear_none
    PK = results.get_matter_power_interpolator()

    idx = k==0
    k[idx] = 1
    out = PK.P(z,k)
    #out *= ((ng*ng*ng)/(rl*rl*rl))
    out[idx] = 0
    return out

def D_plus(OmM,OmL):
    return (5/2)*OmM*(1/(OmM**(4/7) - OmL + (1+0.5*OmM)*(1+(1/70)*OmL)))

def z2a(z):
    return 1/(1+z)

def a2z(a):
    return (1/a) - 1

def da_dtau(a,OmM,OmL):
    da_dtau_2 = 1+OmM*((1/a)-1) + OmL*((a**2) - 1)
    return np.sqrt(da_dtau_2)

def da_dtau__3(a,OmM,OmL):
    da_dtau_2 = 1+OmM*((1/a)-1) + OmL*((a**2) - 1)
    return 1/(np.sqrt(da_dtau_2)**3)

def int_1_da_dtau_3(a,OmM,OmL):
    return scipy.integrate.quad(da_dtau__3,0,a,(OmM,OmL))

def delta(a,OmM,OmL):
    integral = int_1_da_dtau_3(a,OmM,OmL)
    diff = da_dtau(a,OmM,OmL)
    mul = (5*OmM)/(2*a)
    return mul*diff*integral[0]

def dotDelta(a,OmM,OmL):
    return scipy.misc.derivative(delta,a,1e-3,args=(OmM,OmL))

def f(z,OmM,OmL):
    top = OmM*((1+z)**3)
    bottom = OmM*((1+z)**3) - (OmM + OmL - 1)*((1+z)**2) + OmL
    return (top/bottom)**(4/7)

def get_delta_and_dotDelta(z,z1,fname):
    haccparams = read_params(fname)
    cambparams,ns = hacc2camb(haccparams)
    pars = camb.CAMBparams()
    pars.set_cosmology(**cambparams)
    OmB = pars.ombh2 / (pars.h**2)
    OmC = pars.omch2 / (pars.h**2)
    OmNu = pars.omnuh2 / (pars.h**2)
    OmK = pars.omk
    OmM = OmB + OmC
    OmL = 1 - (OmB + OmC + OmNu + OmK)
    d = delta(z2a(z),OmM,OmL)
    d_dot = dotDelta(z2a(z1),OmM,OmL)
    return np.array([d,d_dot],dtype=np.float64)

if __name__ == "__main__":
    k_min = 0
    k_max = 200
    k_bins = 1000
    z = 200
    fname = ""
    if (len(sys.argv) < 3):
        print("USAGE: <file> <outfile> [z] [k_max] [k_min] [k_bins]")
        exit()
    if (len(sys.argv) >= 3):
        fname = sys.argv[1]
        outfile = sys.argv[2]
    if (len(sys.argv) >= 4):
        z = float(sys.argv[3])
    if (len(sys.argv) >= 5):
        k_max = float(sys.argv[4])
    if (len(sys.argv) >= 6):
        k_min = float(sys.argv[5])
    if (len(sys.argv) == 7):
        k_bins = float(sys.argv[6])
    if len(sys.argv) > 7:
        print("USAGE: <file> <outfile> [z] [k_max] [k_min] [k_bins]")
        exit()
    ks = np.linspace(k_min,k_max,k_bins)
    pk = get_pk(z,ks,fname)
    out = np.zeros((len(pk),2),dtype=np.float64)
    out[:,0] = np.linspace(k_min,k_max,k_bins)
    out[:,1] = pk
    out = out.flatten().astype(np.float64)
    out.tofile(outfile)