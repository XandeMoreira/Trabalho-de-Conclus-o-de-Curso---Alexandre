import numpy as np
import math, os, sys
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib.pyplot as plt
#plt.style.use('science'
from scipy.integrate import odeint
from math import pi, cos
from functions import *
from colebank_model import *
import chaospy as cp

# -----------------------------------------------------------------------------

def SolveColebankModel(idcase,krs,krp,kea,kt,plot=False):

    #read the parameters and initial conditions from XLS file
    pars, y0 = ReadParsInit(filename,krs,krp,kea,kt)

    # tempo
    nc = 30
    ns = 15
    T = pars['T']
    N = nc*ns
    tf = nc*T
    t = np.linspace(0, tf, nc*ns)
    k = (nc-1)*ns # indice do ultimo ciclo

    # solve the model
    solution = odeint(Modelo, y0, t, args=(pars,)) #,kEa,kRs,kRp,kT))

    # get solution
    Vsa = solution[:,0]
    Vsv = solution[:,1]
    Vpa = solution[:,2]
    Vpv = solution[:,3]
    Vra = solution[:,4]
    Vla = solution[:,5]
    Vrv = solution[:,6]
    Vlv = solution[:,7]

    # -------------------------------------------------------------------------

    # post-processing
    Era = np.zeros(N)
    Ela = np.zeros(N)
    Erv = np.zeros(N)
    Elv = np.zeros(N)

    psa = Vsa/pars['Csa'] # systemic arteries
    psv = Vsv/pars['Csv'] # systemic veins
    ppa = Vpa/pars['Cpa'] # pulmonary arteries
    ppv = Vpv/pars['Cpv'] # pulmonary veins
    
    # Heart elastance
    for i in range(N):
        tst = t[i] - (t[i] % T)
        Era[i] = ElastanceAtrium(t[i]-tst,pars['EMra'],pars['Emra'],pars['Trra'],pars['tcra'],pars['Tcra'],pars['T']) # Right atrium elastance
        Ela[i] = ElastanceAtrium(t[i]-tst,pars['EMla'],pars['Emla'],pars['Trla'],pars['tcla'],pars['Tcla'],pars['T']) # Left atrium elastance        
        Erv[i] = ElastanceVentricle(t[i]-tst,pars['EMrv'],pars['Emrv'],pars['Tcrv'],pars['Trrv'])     # Right ventricle elastance
        Elv[i] = ElastanceVentricle(t[i]-tst,pars['EMlv'],pars['Emlv'],pars['Tclv'],pars['Trlv'])     # Right ventricle elastance
    
    pra = Era * Vra # Right atrium pressure
    pla = Ela * Vla # Left atrium pressure
    prv = Erv * Vrv # Right ventricle pressure
    plv = Elv * Vlv # Left ventricle pressure


    # salva os dados
    arqs = 'outputs/outputs_%03d' % (idcase)  
    np.savez(arqs, t=t, Vsa=Vsa, Vsv=Vsv, Vpa=Vpa, Vpv=Vpv, 
                        Vra=Vra, Vla=Vla, Vrv=Vrv, Vlv=Vlv,
                        psa=psa, psv=psv, ppa=ppa, ppv=ppv, 
                        pra=pra, pla=pla, prv=prv, plv=plv, 
                        Era=Era, Ela=Ela, Erv=Erv, Elv=Elv)

    qois = CalcOutputs(idcase,k,t,T,psa,ppa,ppv,Vlv,plv,Vrv,prv,pra)
    
    if(plot):
        PlotAll(t[k:],plv[k:],prv[k:],pla[k:],pra[k:],psa[k:],psv[k:],ppa[k:],ppv[k:],path)
        PlotElastances(t[k:],Elv[k:],Ela[k:],Erv[k:],Era[k:],path)
        PlotPVLoops(Vlv[k:],plv[k:],Vrv[k:],prv[k:],path)  
        PlotVolumes(k,t,solution,path)
        PlotPressures(t[k:],plv[k:],psa[k:],psv[k:],ppa[k:],ppv[k:],path)
    
    return qois

# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='file', type=str, help='input filename')
    parser.add_argument('-o', dest='outdir', type=str, help='output directory')
    args = parser.parse_args()

    # normal: kT = kEa = kRs = kRp = 1.0
    # ENMC: kT = 0.75, kEa = 0.70, kRs = 1.0, kRp = 2.75

    # tamanhos
    nsa = 15 # numero de amostras
    ninp = 3  # numero de parametros
    nout = 20 # numero de saidas (qois)

    # distribuicao dos parametros
    d_kEa = cp.Normal(0.70,0.07)
    d_kRp = cp.Normal(2.75,0.275)
    d_kT  = cp.Normal(0.6, 0.06)   # 100 bpm a 60 bpm
    #d_kRs = cp.Normal(2.75,0.275)

    # d_kEa = cp.Normal(0.7, 1e-6)
    # d_kRp = cp.Normal(2.75,1e-6)
    # d_kT = cp.Normal(0.6, 1e-6)

    distribution = cp.J(d_kEa,d_kRp,d_kT)

    samples = distribution.sample(nsa)
    evals = np.empty((nsa,nout))
    
    filename = 'paciente-normotensive.xlsx'
    if(args.file is not None):
        filename = args.file

    path = 'outputs/'
    if(args.outdir is not None):
        path = args.outdir

    # evaluate model at samples
    for i in range(nsa):
        print('caso %d' % i)

        # get sample
        krs,kea,krp,kt = 1.0, samples[0,i], samples[1,i], samples[2,i]

        # solve
        evals[i,:] = SolveColebankModel(i,krs,krp,kea,kt)

    # save data
    arq_inps = path + 'samples'
    arq_outs = path + 'qois'
    np.savez(arq_inps, data=samples)
    np.savez(arq_outs, data=evals)

# end