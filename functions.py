import numpy as np
import math, os, sys
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use('science')
from scipy.integrate import odeint
from math import pi, cos

# -----------------------------------------------------------------------------

#def CalcOutputs(idcase,k,t,T,psa,ppa,ppv,Vlv,plv,Vrv,prv,pra):
def CalcOutputs(k,t,T,psa,ppa,ppv,Vlv,plv,Vrv,prv,pra):
    # Systemic Circulation
    Psa_min = psa[k:].min()
    Psa_max = psa[k:].max()
    Psa_mean = (1.0/T)*np.trapz(psa[k:],t[k:])
    print(' Psa_min  = %.2f' % Psa_min)
    print(' Psa_max  = %.2f' % Psa_max)
    print(' Psa_mean = %.2f' % Psa_mean)

    #Pulmonary Circulation
    Pmean = (1.0/T)*np.trapz(ppa[k:],t[k:])
    Pmin = ppa[k:].min()
    Pmax = ppa[k:].max()
    PPVmean = (1.0/T)*np.trapz(ppv[k:],t[k:])
    print(' PAP_mean = %.2f' % Pmean)
    print(' PAP_min = %.2f' % Pmin)
    print(' PAP_max = %.2f' % Pmax)
    print(' PPV_mean = %.2f' % PPVmean)

    #Left Ventricle
    LV_EDV = Vlv[k:].max()
    LV_ESV = Vlv[k:].min()
    LV_SV = LV_EDV-LV_ESV
    LV_EF = (LV_SV/LV_EDV)*100
    LV_Pmax = plv[k:].max()
    print(' LV_EDV   = %.2f' % LV_EDV)
    print(' LV_ESV   = %.2f' % LV_ESV)
    print(' LV_SV   = %.2f' % LV_SV)
    print(' LV_EF    = %.2f' % LV_EF)
    print(' LV_Pmax  = %.2f' % LV_Pmax)

    # Right Ventricule
    RV_EDV = Vrv[k:].max()
    RV_ESV = Vrv[k:].min()
    RV_SV = RV_EDV - RV_ESV
    RV_EF = (RV_SV/RV_EDV)*100
    RV_Pmax = prv[k:].max()
    RV_Pmin = prv[k:].min()
    print(' RV_EDV   = %.2f' % RV_EDV)
    print(' RV_ESV   = %.2f' % RV_ESV)
    print(' RV_SV   = %.2f' % RV_SV)
    print(' RV_EF    = %.2f' % RV_EF)
    print(' RV_Pmax  = %.2f' % RV_Pmax)
    print(' RV_Pmin  = %.2f' % RV_Pmin)

#QOI's excedentes
    #right atrium
    Pra_min = pra[k:].min()
    Pra_max = pra[k:].max()
    Pra_mean = (1.0/T)*np.trapz(pra[k:],t[k:])
    print(' Pra_min  = %.2f' % Pra_min)
    print(' Pra_max  = %.2f' % Pra_max)
    print(' Pra_mean = %.2f' % Pra_mean)

#Cardiac Output
    
    arqs_data = 'outputs/qois_%03d' #% (idcase)
    qois = np.transpose([Psa_min,Psa_max,Psa_mean,Pmean,Pmin,Pmax,PPVmean,
                         LV_EDV,LV_ESV,LV_SV,LV_EF,LV_Pmax,
                         RV_EDV,RV_ESV,RV_SV,RV_EF,RV_Pmax,Pra_min,Pra_max,Pra_mean])
    np.savez(arqs_data, qois=qois)
    return qois

# -----------------------------------------------------------------------------

def PlotAll(t,plv,prv,pla,pra,psa,psv,ppa,ppv,path):
    
    fig, axs = plt.subplots(3, 3)

    # linha 1
    axs[0, 0].plot(t, pla)
    axs[0, 0].set(xlabel='time [s]', ylabel='pla')

    axs[0, 1].plot(t, plv, 'tab:orange')
    axs[0, 1].set(xlabel='time [s]', ylabel='plv')

    axs[0, 2].plot(t, psa, 'tab:orange')
    axs[0, 2].set(xlabel='time [s]', ylabel='psa')
    
    # linha 2
    axs[1, 0].plot(t, psv, 'tab:green')
    axs[1, 0].set(xlabel='time [s]', ylabel='psv')
    
    axs[1, 1].plot(t, pra, 'tab:red')
    axs[1, 1].set(xlabel='time [s]', ylabel='pra')

    axs[1, 2].plot(t, prv, 'tab:red')
    axs[1, 2].set(xlabel='time [s]', ylabel='prv')

    # linha 3
    axs[2, 0].plot(t, ppa, 'tab:green')
    axs[2, 0].set(xlabel='time [s]', ylabel='ppa')
    
    axs[2, 1].plot(t, ppv, 'tab:red')
    axs[2, 1].set(xlabel='time [s]', ylabel='ppv')

    #axs[2, 1].plot(t, psa, 'tab:red')
    #axs[2, 2].set_title('Axis [2, 2]')

    plt.tight_layout()
    plt.savefig(path+'fig_pressures.pdf', format='pdf', dpi=300)
    plt.savefig(path+'fig_pressures.png', format='png', dpi=300)
    plt.show()

# -----------------------------------------------------------------------------

def PlotElastances(t,Elv,Ela,Erv,Era,path):
    plt.plot(t, Elv, label ='Elv')
    plt.plot(t, Ela, label ='Ela')
    plt.plot(t, Erv, label ='Erv')
    plt.plot(t, Era, label ='Era')
    plt.legend(loc='best')
    plt.xlabel('time [s]')
    plt.ylabel('elastance')
    plt.tight_layout()
    plt.savefig(path+'fig_elastance.png', format='png', dpi=300)
    plt.show()

# -----------------------------------------------------------------------------

def PlotPressures(t,plv,prv,psa,psv,ppa,ppv,path):
    
    plt.plot(t,plv, label ='Plv', color = 'red')
    plt.xlabel('tempo [s]')
    plt.ylabel('pressão [mmHg]')
    plt.title(r'$P_{lv}$')
    plt.tight_layout()
    plt.savefig(path+'fig_plv.pdf', format='pdf', dpi=300)
    plt.show()

    plt.plot(t,prv, label ='Prv', color = 'red')
    plt.xlabel('tempo [s]')
    plt.ylabel('pressão [mmHg]')
    plt.title(r'$P_{rv}$')
    plt.tight_layout()
    plt.savefig(path+'fig_prv.pdf', format='pdf', dpi=300)
    plt.show()

    plt.plot(t, psa, label='Psa', color = 'red') 
    plt.xlabel('tempo [s]')
    plt.ylabel('pressão [mmHg]')
    plt.title(r'$P_{sa}$')
    plt.tight_layout()
    plt.savefig(path+'fig_psa.pdf', format='pdf', dpi=300)
    plt.show()

    plt.plot(t, psv, label='Psv')
    plt.xlabel('time [s]')
    plt.ylabel('pressure [mmHg]')
    plt.tight_layout()
    plt.savefig(path+'fig_psv.pdf', format='pdf', dpi=300)
    plt.show()

    plt.plot(t, ppa, label='Ppa')
    plt.xlabel('time [s]')
    plt.ylabel('pressure [mmHg]')
    plt.tight_layout()
    plt.savefig(path+'fig_ppa.pdf', format='pdf', dpi=300)
    plt.show()

    plt.plot(t, ppv, label='Ppv')
    plt.xlabel('time [s]')
    plt.ylabel('pressure [mmHg]')
    plt.tight_layout()
    plt.savefig(path+'fig_ppv.pdf', format='pdf', dpi=300)
    plt.show()
# -----------------------------------------------------------------------------

def PlotVolumes(k,t,solution,path):
        
    plt.plot(t[k:], solution[k:,0], label='Vsa', color ='red')
    #plt.legend(loc='best')
    plt.xlabel('tempo [s]')
    plt.ylabel('volume [mL]')
    plt.title(r'$V_{sa}$')
    plt.savefig(path+'fig_vsa.pdf', format='pdf', dpi=300)
    plt.show()

    plt.plot(t[k:], solution[k:,1], label='Vsv')
    plt.legend(loc='best')
    plt.xlabel('time [s]')
    plt.ylabel('volume [mL]')
    plt.savefig(path+'fig_vsv.pdf', format='pdf', dpi=300)
    plt.close()

    plt.plot(t[k:], solution[k:,2], label='Vpa')
    plt.legend(loc='best')
    plt.xlabel('time [s]')
    plt.ylabel('volume [mL]')
    plt.savefig(path+'fig_vpa.pdf', format='pdf', dpi=300)
    plt.close()

    plt.plot(t[k:], solution[k:,3], label='Vpv')
    plt.legend(loc='best')
    plt.xlabel('time [s]')
    plt.ylabel('volume [mL]')
    plt.savefig(path+'fig_vpv.pdf', format='pdf', dpi=300)
    plt.close()
    

    plt.plot(t[k:], solution[k:,4], label='Vra')
    plt.legend(loc='best')
    plt.xlabel('time [s]')
    plt.ylabel('volume [mL]')
    plt.savefig(path+'fig_vra.pdf', format='pdf', dpi=300)
    plt.close()
    

    plt.plot(t[k:], solution[k:,5], label='Vla')
    plt.legend(loc='best')
    plt.xlabel('time [s]')
    plt.ylabel('volume [mL]')
    plt.savefig(path+'fig_vla.pdf', format='pdf', dpi=300)
    plt.close()

    plt.plot(t[k:], solution[k:,6], label='Vrv', color = 'red')
    #plt.legend(loc='best')
    plt.xlabel('tempo [s]')
    plt.ylabel('volume [mL]')
    plt.title(r'$V_{rv}$')
    plt.savefig(path+'fig_vrv.pdf', format='pdf', dpi=300)
    plt.show()

    plt.plot(t[k:], solution[k:,7], label='Vlv', color = 'red')
    #plt.legend(loc='best')
    plt.xlabel('tempo [s]')
    plt.ylabel('volume [mL]')
    plt.title(r'$V_{lv}$')
    plt.savefig(path+'fig_vlv.pdf', format='pdf', dpi=300)
    plt.show()

# -----------------------------------------------------------------------------

def PlotPVLoops(Vlv,plv,Vrv,prv,path):

    plt.plot(Vrv, prv, color ='red')
    plt.xlabel('Vrv [mL]')
    plt.ylabel('Prv [mmHg]')
    plt.tight_layout()
    plt.savefig(path+'fig_pvloop_rv.pdf', format='pdf', dpi=300)
    plt.show()

    plt.plot(Vlv, plv, color ='red')
    plt.xlabel('Vlv [mL]')
    plt.ylabel('Plv[mmHg]')
    plt.tight_layout()
    plt.savefig(path+'fig_pvloop_lv.pdf', format='pdf', dpi=300)
    plt.show()

