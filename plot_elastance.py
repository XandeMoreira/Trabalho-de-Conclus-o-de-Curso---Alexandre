import numpy as np
import math, sys
from scipy.integrate import odeint
from math import pi, cos
import matplotlib.pyplot as plt
import pandas as pd

def ElastanceAtrium(t,EaM,Eam,Tar,tac,Tac,T):
    if t<=Tar:
        Ea = (EaM-Eam)*(1-cos(pi*(t-Tar)/(T-Tac+Tar)))/2+Eam
    elif t <= tac:
        Ea = Eam
    elif t <= Tac:
        Ea = (EaM-Eam)*(1-cos(pi*(t-tac)/(Tac-tac)))/2+Eam
    else:
        Ea = (EaM-Eam)*(1+cos(pi*(t-Tac)/(T-Tac+Tar)))/2+Eam
    return Ea


def ElastanceVentricle(t,EvM,Evm,Tvc,Tvr):
    if t<=Tvc:
        Elv = (EvM-Evm)*(1-cos(pi*t/Tvc))/2 + Evm
    elif t <= Tvr:
        Elv = (EvM-Evm)*(1+cos(pi*(t-Tvc)/(Tvr-Tvc)))/2 + Evm
    else:
        Elv = Evm  
    return Elv


if __name__ == "__main__":

    kEa = 1.0
    kT = 0.60
    kR = 1.0

    #dados = pd.read_excel('paciente1.xlsx') 
    dados = pd.read_excel('paciente-normotensive.xlsx')  
    pars = dados['pars']

    EMra = kEa * pars[12]
    Emra = kEa * pars[13]
    EMla = kEa * pars[14]
    Emla = kEa * pars[15]

    EMrv = kEa * pars[16]
    Emrv = kEa * pars[17]
    EMlv = kEa * pars[18]
    Emlv = kEa * pars[19]

    Trra = kT * pars[20]
    tcra = kT * pars[21]
    Tcra = kT * pars[22]
    Tcrv = kT * pars[23]
    Trrv = kT * pars[24]

    T = 1.0*kT

    # time cycle
    nc = 1
    ns = 1000
    N = nc*ns
    tf = nc*T
    t = np.linspace(0, tf, nc*ns)
    k = (nc-1)*ns

    #Vetores para as elastancias
    Era = np.zeros(N)
    Ela = np.zeros(N)
    Erv = np.zeros(N)
    Elv = np.zeros(N)

    # Timing
    Trra = Trra         # end right atrium relaxation
    tcra = Trra + tcra  # right atrium begin contraction
    Tcra = tcra + Tcra  # right atrium end contraction

    Trla = Trra*(1.01)  # end left atrium relaxation
    tcla = tcra*(1.05)  # left atrium begin contraction
    Tcla = Tcra         # left atrium end contraction

    Tcrv = Tcrv         # right ventricle contraction
    Trrv = Tcrv + Trrv  # right ventricle relaxation
        
    Tclv = Tcrv*(0.95)  # left ventricle contraction
    Trlv = Trrv         # left ventricle relaxation


    # Heart chamber elastance function (Elv,Ela,Erv, Era )
    for i in range(N):
        tst = t[i] - (t[i] % T)
        #print(t[i],tst,t[i]%T)
        Era[i] = ElastanceAtrium(t[i]-tst,EMra,Emra,Trra,tcra,Tcra,T) # Right atrium elastance
        Ela[i] = ElastanceAtrium(t[i]-tst,EMla,Emla,Trla,tcla,Tcla,T) # Left atrium elastance        
        Erv[i] = ElastanceVentricle(t[i]-tst,EMrv,Emrv,Tcrv,Trrv)     # Right ventricle elastance
        Elv[i] = ElastanceVentricle(t[i]-tst,EMlv,Emlv,Tclv,Trlv)     # Right ventricle elastance


    #Gráficos para as elastâncias
    plt.plot(t[k:],Ela[k:], label ='Ela', color='red')
    plt.plot(t[k:],Elv[k:], label ='Elv', color='blue')

    plt.axvline(x = Trla, color = 'red', linestyle ='--')
    plt.axvline(x = tcla, color = 'red', linestyle ='--')
    plt.axvline(x = Tcla, color = 'red', linestyle ='--')
    plt.axvline(x = Tclv, color = 'blue', linestyle ='--')
    plt.axvline(x = Trlv, color = 'blue', linestyle ='--')
    
    # Adicionando as legendas ao lado das linhas verticais
    for x, text in zip([Trla, tcla, Tcla, Tclv, Trlv], ['Trla          ', '          tcla', '         Tcla', '          Tclv', '          Trlv']):
        plt.text(x, max(Ela) + 0.05, text, color='black', verticalalignment='bottom', horizontalalignment='center')


    plt.legend(loc='best')
    plt.xlabel('Tempo [s]')
    plt.ylabel('E [mmHg ml⁻¹]')
    plt.savefig('ElastanceL.pdf', format='pdf', dpi = 300)
    plt.show()

    #Gráficos para as elastâncias 

    plt.plot(t[k:],Era[k:], label ='Era', color='red')
    plt.plot(t[k:],Erv[k:], label ='Erv',color='blue')

    plt.axvline(x = Trra, color = 'red', linestyle ='--')
    plt.axvline(x = tcra, color = 'red', linestyle ='--')
    plt.axvline(x = Tcra, color = 'red', linestyle ='--')
    plt.axvline(x = Tcrv, color = 'blue', linestyle ='--')
    plt.axvline(x = Trrv, color = 'blue', linestyle ='--')

    # Adicionando as legendas ao lado das linhas verticais
    for x, text in zip([Trra, tcra, Tcra, Tcrv, Trrv], ['Trra          ', '          tcra', '         Tcra', '          Tcrv', '          Trrv']):
        plt.text(x, max(Ela) + 0.05, text, color='black', verticalalignment='bottom', horizontalalignment='center')


    plt.legend(loc='best')
    plt.xlabel('Tempo [s]')
    plt.ylabel('E [mmHg ml⁻¹]')
    plt.savefig('ElastanceR.pdf', format='pdf', dpi = 300)
    plt.show()

    print('RA',EMra,Emra,Trra,tcra,Tcra,T) # Right atrium elastance
    print('LA',EMla,Emla,Trla,tcla,Tcla,T) # Left atrium elastance        
    print('RV',EMrv,Emrv,Tcrv,Trrv)     # Right ventricle elastance
    print('LV',EMlv,Emlv,Tclv,Trlv)     # Right ventricle elastance
