import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import*
from colebank_model import *
import pylab as pl
import scienceplots
plt.style.use('nature')

# definitions
nsa  = 15
nqoi = 20

# timings
nc = 30
ns = 15
N = nc*ns
k = (nc-1)*ns # indice do ultimo ciclo

# files
arqb_out = 'outputs/outputs_'
arqb_qoi = 'outputs/qois_'
qois_str = ['psa_min','psa_max','psa_mean','pap_mean','pap_min','pap_max','ppv_mean',
            'lv_edv','lv_esv','lv_sv','lv_ef','lv_pmax','rv_edv','rv_esv','rv_sv','rv_ef','rv_pmax','pra_min','pra_max','pra_mean']

# load QOIs
mat_qoi = np.zeros((nsa,nqoi))
for j in range(nsa):
    # arq = arqb + '%03d.txt' % (j)
    # qoi = np.loadtxt(arq)
    arq = arqb_qoi + '%03d.npz' % (j)
    d = np.load(arq)
    qoi = d['qois']
    mat_qoi[j,:] = qoi[:]
    
# print(mat_qoi)
print('quantities of interest')
for j in range(nqoi):
    mu = np.mean( mat_qoi[:,j] )
    std = np.std( mat_qoi[:,j] )
    print(' %s   \t %.4f \t %.4f' % (qois_str[j],mu,std))


# criar um dataframe para salvar as 20 QoI's
def create_dataframe ():
    data = {'Média': [np.mean(mat_qoi[:,0]),np.mean(mat_qoi[:,1]),np.mean(mat_qoi[:,2]),np.mean(mat_qoi[:,3]),np.mean(mat_qoi[:,4] ),np.mean(mat_qoi[:,5]),np.mean(mat_qoi[:,6]),np.mean(mat_qoi[:,7]),np.mean(mat_qoi[:,8]),np.mean(mat_qoi[:,9]),np.mean(mat_qoi[:,10]),np.mean(mat_qoi[:,11]),np.mean(mat_qoi[:,12]),np.mean(mat_qoi[:,13]),np.mean(mat_qoi[:,14]),np.mean(mat_qoi[:,15]),np.mean(mat_qoi[:,16]),np.mean(mat_qoi[:,17]),np.mean(mat_qoi[:,18]),np.mean(mat_qoi[:,19])], 'Desvio Padrão':[np.std(mat_qoi[:,0]),np.std(mat_qoi[:,1]),np.std(mat_qoi[:,2]),np.std(mat_qoi[:,3]),np.std(mat_qoi[:,4] ),np.std(mat_qoi[:,5]),np.std(mat_qoi[:,6]),np.std(mat_qoi[:,7]),np.std(mat_qoi[:,8]),np.std(mat_qoi[:,9]),np.std(mat_qoi[:,10]),np.std(mat_qoi[:,11]),np.std(mat_qoi[:,12]),np.std(mat_qoi[:,13]),np.std(mat_qoi[:,14]),np.std(mat_qoi[:,15]),np.std(mat_qoi[:,16]),np.std(mat_qoi[:,17]),np.std(mat_qoi[:,18]),np.std(mat_qoi[:,19])]}
    results_qoi = pd.DataFrame(data, index = ['psa_min','psa_max','psa_mean','pap_mean','pap_min','pap_max','ppv_mean','lv_edv','lv_esv','lv_sv','lv_ef','lv_pmax','rv_edv','rv_esv','rv_sv','rv_ef','rv_pmax','pra_min','pra_max','pra_mean'])
    results_qoi.to_excel('15QoI.xlsx', index=True)
create_dataframe ()

# Graficos para analises
psa_mat = np.zeros((nsa,ns))
psa_mean = np.zeros(ns)
psa_std = np.zeros(ns)
vsa_mat = np.zeros((nsa,ns))
vsa_mean = np.zeros(ns)
vsa_std = np.zeros(ns)

psv_mat = np.zeros((nsa,ns))
psv_mean = np.zeros(ns)
psv_std = np.zeros(ns)
vsv_mat = np.zeros((nsa,ns))
vsv_mean = np.zeros(ns)
vsv_std = np.zeros(ns)

ppa_mat = np.zeros((nsa,ns))
ppa_mean = np.zeros(ns)
ppa_std = np.zeros(ns)
vpa_mat = np.zeros((nsa,ns))
vpa_mean = np.zeros(ns)
vpa_std = np.zeros(ns)

ppv_mat = np.zeros((nsa,ns))
ppv_mean = np.zeros(ns)
ppv_std = np.zeros(ns)
vpv_mat = np.zeros((nsa,ns))
vpv_mean = np.zeros(ns)
vpv_std = np.zeros(ns)

pra_mat = np.zeros((nsa,ns))
pra_mean = np.zeros(ns)
pra_std = np.zeros(ns)
vra_mat = np.zeros((nsa,ns))
vra_mean = np.zeros(ns)
vra_std = np.zeros(ns)

pla_mat = np.zeros((nsa,ns))
pla_mean = np.zeros(ns)
pla_std = np.zeros(ns)
vla_mat = np.zeros((nsa,ns))
vla_mean = np.zeros(ns)
vla_std = np.zeros(ns)

prv_mat = np.zeros((nsa,ns))
prv_mean = np.zeros(ns)
prv_std = np.zeros(ns)
vrv_mat = np.zeros((nsa,ns))
vrv_mean = np.zeros(ns)
vrv_std = np.zeros(ns)

plv_mat = np.zeros((nsa,ns))
plv_mean = np.zeros(ns)
plv_std = np.zeros(ns)
vlv_mat = np.zeros((nsa,ns))
vlv_mean = np.zeros(ns)
vlv_std = np.zeros(ns)

for i in range(nsa):
    # arqf = arqb + '%03d.txt' % (i)
    # t,Vsa,Vsv,Vpa,Vpv,Vra,Vla,Vrv,Vlv,psa,psv,ppa,ppv,pra,pla,prv,plv,Era,Ela,Erv,Elv = np.loadtxt(arqf, skiprows = 1, usecols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],unpack = True)
    
    arqf = arqb_out + '%03d.npz' % (i)
    d = np.load(arqf)
    t = d['t']
    Vsa, Vsv, Vpa, Vpv = d['Vsa'], d['Vsv'], d['Vpa'], d['Vpv']
    Vra, Vla, Vrv, Vlv = d['Vra'], d['Vla'], d['Vrv'], d['Vlv']
    psa, psv, ppa, ppv = d['psa'], d['psv'], d['ppa'], d['ppv']
    pra, pla, prv, plv = d['pra'], d['pla'], d['prv'], d['plv']
    Era, Ela, Erv, Elv = d['Era'], d['Ela'], d['Erv'], d['Elv']
    
    psa_mat[i,:] = psa[k:]
    psv_mat[i,:] = psv[k:]
    ppa_mat[i,:] = ppa[k:]
    ppv_mat[i,:] = ppv[k:]
    pra_mat[i,:] = pra[k:]
    pla_mat[i,:] = pla[k:]
    prv_mat[i,:] = prv[k:]
    plv_mat[i,:] = plv[k:]

    vsa_mat[i,:] = Vsa[k:]
    vsv_mat[i,:] = Vsv[k:]
    vpa_mat[i,:] = Vpa[k:]
    vpv_mat[i,:] = Vpv[k:]
    vra_mat[i,:] = Vra[k:]
    vla_mat[i,:] = Vla[k:]
    vrv_mat[i,:] = Vrv[k:]
    vlv_mat[i,:] = Vlv[k:]

for i in range(ns):

    psa_mean[i] = np.mean(psa_mat[:,i])
    psa_std[i] = np.std(psa_mat[:,i])
    vsa_mean[i] = np.mean(vsa_mat[:,i])
    vsa_std[i] = np.std(vsa_mat[:,i])

    psv_mean[i] = np.mean(psv_mat[:,i])
    psv_std[i] = np.std(psv_mat[:,i])
    vsv_mean[i] = np.mean(vsv_mat[:,i])
    vsv_std[i] = np.std(vsv_mat[:,i])

    ppa_mean[i] = np.mean(ppa_mat[:,i])
    ppa_std[i] = np.std(ppa_mat[:,i])
    vpa_mean[i] = np.mean(vpa_mat[:,i])
    vpa_std[i] = np.std(vpa_mat[:,i])

    ppv_mean[i] = np.mean(ppv_mat[:,i])
    ppv_std[i] = np.std(ppv_mat[:,i])
    vpv_mean[i] = np.mean(vpv_mat[:,i])
    vpv_std[i] = np.std(vpv_mat[:,i])

    pra_mean[i] = np.mean(pra_mat[:,i])
    pra_std[i] = np.std(pra_mat[:,i])
    vra_mean[i] = np.mean(vra_mat[:,i])
    vra_std[i] = np.std(vra_mat[:,i])

    pla_mean[i] = np.mean(pla_mat[:,i])
    pla_std[i] = np.std(pla_mat[:,i])
    vla_mean[i] = np.mean(vla_mat[:,i])
    vla_std[i] = np.std(vla_mat[:,i])

    prv_mean[i] = np.mean(prv_mat[:,i])
    prv_std[i] = np.std(prv_mat[:,i])
    vrv_mean[i] = np.mean(vrv_mat[:,i])
    vrv_std[i] = np.std(vrv_mat[:,i])


    plv_mean[i] = np.mean(plv_mat[:,i])
    plv_std[i] = np.std(plv_mat[:,i])
    vlv_mean[i] = np.mean(vlv_mat[:,i])
    vlv_std[i] = np.std(vlv_mat[:,i])

#plot para as pressões (fill_between)
fig, axs = plt.subplots(2, 4)
axs[0, 0].plot(t[k:], psa_mean, color='darkgreen', lw=2)
axs[0, 0].fill_between(t[k:], psa_mean - psa_std, psa_mean + psa_std, alpha=0.25, color='darkgreen')
axs[0, 0].set_title('Psa')
axs[0, 0].set(xlabel='Time [s]', ylabel='Psa [mmHg]')

axs[0, 1].plot(t[k:], psv_mean, color='darkgreen', lw=2)
axs[0, 1].fill_between(t[k:], psv_mean - psv_std, psv_mean + psv_std, alpha=0.25, color='darkgreen')
axs[0, 1].set_title('Psv')
axs[0, 1].set(xlabel='Time [s]', ylabel='Psv [mmHg]')

axs[0, 2].plot(t[k:],ppa_mean, color='darkgreen', lw=2)
axs[0, 2].fill_between(t[k:], ppa_mean - ppa_std, ppa_mean + ppa_std, alpha=0.25, color='darkgreen')
axs[0, 2].set_title('Ppa')
axs[0, 2].set(xlabel='Time [s]', ylabel='Ppa [mmHg]')
    
axs[0, 3].plot(t[k:], ppv_mean, color='darkgreen',lw=2)
axs[0, 3].fill_between(t[k:], ppv_mean - ppv_std, ppv_mean + ppv_std, alpha=0.25, color='darkgreen')
axs[0, 3].set_title('Ppv')
axs[0, 3].set(xlabel='Time [s]', ylabel='Ppv [mmHg]')
    
axs[1, 0].plot(t[k:], pra_mean, color='darkgreen',lw=2)
axs[1, 0].fill_between(t[k:], pra_mean - pra_std, pra_mean + pra_std, alpha=0.25, color='darkgreen')
axs[1, 0].set_title('Pra')
axs[1, 0].set(xlabel='Time [s]', ylabel='Pra [mmHg]')
        
axs[1, 1].plot(t[k:], pla_mean, color='darkgreen',lw=2)
axs[1, 1].fill_between(t[k:], pla_mean - pla_std, pla_mean + pla_std, alpha=0.25, color='darkgreen')
axs[1, 1].set_title('Pla')
axs[1, 1].set(xlabel='Time [s]', ylabel='Pla [mmHg]')

axs[1, 2].plot(t[k:], prv_mean, color='darkgreen',lw=2)
axs[1, 2].fill_between(t[k:], prv_mean - prv_std, prv_mean + prv_std, alpha=0.25, color='darkgreen')
axs[1, 2].set_title('Prv')
axs[1, 2].set(xlabel='Time [s]', ylabel='Prv [mmHg]')

axs[1, 3].plot(t[k:], plv_mean, color='darkgreen',lw=2)
axs[1, 3].fill_between(t[k:], plv_mean - plv_std, plv_mean + plv_std, alpha=0.25, color='darkgreen')
axs[1, 3].set_title('Plv')
axs[1, 3].set(xlabel='Time [s]', ylabel='Plv [mmHg]')

plt.tight_layout()
plt.savefig('save_figures/pressoes.png',format = 'png',dpi=300)
plt.show()

# plot para volumes (fillbetween)
fig, axs = plt.subplots(2, 4)
axs[0, 0].plot(t[k:], vsa_mean, color='darkgreen', lw=2)
axs[0, 0].fill_between(t[k:], vsa_mean - vsa_std, vsa_mean + vsa_std, alpha=0.25, color='darkgreen')
axs[0, 0].set_title('Vsa')
axs[0, 0].set(xlabel='Time [s]', ylabel='Vsa [mL]')

axs[0, 1].plot(t[k:], vsv_mean, color='darkgreen', lw=2)
axs[0, 1].fill_between(t[k:], vsv_mean - vsv_std, vsv_mean + vsv_std, alpha=0.25, color='darkgreen')
axs[0, 1].set_title('Vsv')
axs[0, 1].set(xlabel='Time [s]', ylabel='Vsv [mL]')

axs[0, 2].plot(t[k:],vpa_mean, color='darkgreen', lw=2)
axs[0, 2].fill_between(t[k:], vpa_mean - vpa_std, vpa_mean + vpa_std, alpha=0.25, color='darkgreen')
axs[0, 2].set_title('Vpa')
axs[0, 2].set(xlabel='Time [s]', ylabel='Vpa [mL]')
    
axs[0, 3].plot(t[k:], vpv_mean, color='darkgreen',lw=2)
axs[0, 3].fill_between(t[k:], vpv_mean - vpv_std, vpv_mean + vpv_std, alpha=0.25, color='darkgreen')
axs[0, 3].set_title('Vpv')
axs[0, 3].set(xlabel='Time [s]', ylabel='Vpv [mL]')
    
axs[1, 0].plot(t[k:], vra_mean, color='darkgreen',lw=2)
axs[1, 0].fill_between(t[k:], vra_mean - vra_std, vra_mean + vra_std, alpha=0.25, color='darkgreen')
axs[1, 0].set_title('Vra')
axs[1, 0].set(xlabel='Time [s]', ylabel='Vra [mL]')
        
axs[1, 1].plot(t[k:], vla_mean, color='darkgreen',lw=2)
axs[1, 1].fill_between(t[k:], vla_mean - vla_std, vla_mean + vla_std, alpha=0.25, color='darkgreen')
axs[1, 1].set_title('Vla')
axs[1, 1].set(xlabel='Time [s]', ylabel='Vla [mL]')

axs[1, 2].plot(t[k:], vrv_mean, color='darkgreen',lw=2)
axs[1, 2].fill_between(t[k:], vrv_mean - vrv_std, vrv_mean + vrv_std, alpha=0.25, color='darkgreen')
axs[1, 2].set_title('Vrv')
axs[1, 2].set(xlabel='Time [s]', ylabel='Vrv [mL]')

axs[1, 3].plot(t[k:], vlv_mean, color='darkgreen',lw=2)
axs[1, 3].fill_between(t[k:], vlv_mean - vlv_std, vlv_mean + vlv_std, alpha=0.25, color='darkgreen')
axs[1, 3].set_title('Vlv')
axs[1, 3].set(xlabel='Time [s]', ylabel='Vlv [mL]')

plt.tight_layout()
plt.savefig('save_figures/volumes.png',format = 'png',dpi=300)
plt.show()

# plot para os pv_loops
# plot para os pv_loops
fig, axs = plt.subplots(2, 2)
 
axs[0, 0].plot(vra_mean, pra_mean, color='darkgreen',lw=2)
axs[0, 0].plot(np.percentile(vra_mat,10, axis=0), np.percentile(pra_mat,10,axis=0), "--",color='darkgreen',lw=2)
axs[0, 0].plot(np.percentile(vra_mat,90, axis=0), np.percentile(pra_mat,90,axis=0), ":",color='darkgreen',lw=2)
axs[0, 0].set(xlabel='Vra [mL]', ylabel='Pra [mmHg]')
        
axs[0, 1].plot(vla_mean, pla_mean, color='darkgreen',lw=2)
axs[0, 1].plot(np.percentile(vla_mat,10, axis=0), np.percentile(pla_mat,10,axis=0), "--",color='darkgreen',lw=2)
axs[0, 1].plot(np.percentile(vla_mat,90, axis=0), np.percentile(pla_mat,90,axis=0), ":",color='darkgreen',lw=2)
axs[0, 1].set(xlabel='Vla [mL]', ylabel='Pla [mmHg]')

axs[1, 0].plot(vrv_mean, prv_mean, color='darkgreen',lw=2)
axs[1, 0].plot(np.percentile(vrv_mat,10, axis=0), np.percentile(prv_mat,10,axis=0), "--",color='darkgreen',lw=2)
axs[1, 0].plot(np.percentile(vrv_mat,90, axis=0), np.percentile(prv_mat,90,axis=0), ":",color='darkgreen',lw=2)
axs[1, 0].set(xlabel='Vrv [mL]', ylabel='Prv [mmHg]')

axs[1, 1].plot(vlv_mean, plv_mean, color='darkgreen',lw=2)
axs[1, 1].plot(np.percentile(vlv_mat,10, axis=0), np.percentile(plv_mat,10,axis=0), "--",color='darkgreen',lw=2)
axs[1, 1].plot(np.percentile(vlv_mat,90, axis=0), np.percentile(plv_mat,90,axis=0), ":",color='darkgreen',lw=2)
axs[1, 1].set(xlabel='Vlv [mL]', ylabel='Plv [mmHg]')

plt.tight_layout()
plt.savefig('save_figures/Pv_loops.png',format = 'png',dpi=300)
plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------#
# Plotagens Individuais (para salvarmos os gráficos, caso queiramos)
plt.plot(t[k:],psa_mean,lw=2,label = 'Psa')
plt.fill_between(t[k:], psa_mean - psa_std, psa_mean + psa_std, alpha=0.5)
plt.legend(loc='best')
plt.xlabel('Time [s]')
plt.ylabel('Psa [mmHg]')
plt.tight_layout()
plt.savefig('save_figures/Psa.png',format = 'png',dpi=300)
plt.show()

plt.plot(t[k:],psv_mean,lw=2,label = 'Psv')
plt.fill_between(t[k:], psv_mean - psv_std, psv_mean + psv_std, alpha=0.5)
plt.legend(loc='best')
plt.xlabel('Time [s]')
plt.ylabel('Psv [mmHg]')
plt.tight_layout()
plt.savefig('save_figures/Psv.png',format = 'png',dpi=300)
plt.show()

plt.plot(t[k:],ppa_mean,lw=2,label = 'Ppa')
plt.fill_between(t[k:], ppa_mean - ppa_std, ppa_mean + ppa_std, alpha=0.5)
plt.legend(loc='best')
plt.xlabel('Time [s]')
plt.ylabel('Ppa [mmHg]')
plt.tight_layout()
plt.savefig('save_figures/Ppa.png',format = 'png',dpi=300)
plt.show()

plt.plot(t[k:],ppv_mean,lw=2,label = 'Ppv')
plt.fill_between(t[k:], ppv_mean - ppv_std, ppv_mean + ppv_std, alpha=0.5)
plt.legend(loc='best')
plt.xlabel('Time [s]')
plt.ylabel('Ppv [mmHg]')
plt.tight_layout()
plt.savefig('save_figures/Ppv.png',format = 'png',dpi=300)
plt.show()

plt.plot(t[k:],pra_mean,lw=2,label = 'Pra')
plt.fill_between(t[k:], pra_mean - pra_std, pra_mean + pra_std, alpha=0.5)
plt.legend(loc='best')
plt.xlabel('Time [s]')
plt.ylabel('Pra [mmHg]')
plt.tight_layout()
plt.savefig('save_figures/Pra.png',format = 'png',dpi=300)
plt.show()

plt.plot(t[k:],pla_mean,lw=2,label = 'Pla')
plt.fill_between(t[k:], pla_mean - pla_std, pla_mean + pla_std, alpha=0.5)
plt.legend(loc='best')
plt.xlabel('Time [s]')
plt.ylabel('Pla [mmHg]')
plt.tight_layout()
plt.savefig('save_figures/Pla.png',format = 'png',dpi=300)
plt.show()

plt.plot(t[k:],prv_mean,lw=2,label = 'Prv')
plt.fill_between(t[k:], prv_mean - prv_std, prv_mean + prv_std, alpha=0.5)
plt.legend(loc='best')
plt.xlabel('Time [s]')
plt.ylabel('Prv [mmHg]')
plt.tight_layout()
plt.savefig('save_figures/Prv.png',format = 'png',dpi=300)
plt.show()

plt.plot(t[k:],plv_mean,lw=2,label = 'Plv')
plt.fill_between(t[k:], plv_mean - plv_std, plv_mean + plv_std, alpha=0.5)
plt.legend(loc='best')
plt.xlabel('Time [s]')
plt.ylabel('Plv [mmHg]')
plt.tight_layout()
plt.savefig('save_figures/Plv.png',format = 'png',dpi=300)
plt.show()
sys.exit(0)