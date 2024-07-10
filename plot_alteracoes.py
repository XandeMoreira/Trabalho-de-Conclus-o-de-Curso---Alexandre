import numpy as np
import matplotlib.pyplot as plt
import sys
import os


# Diretórios dos arquivos
dir_normal = 'NORMAL2'
dir_covid = 'COVID'

# Nomes dos arquivos
file_normal = 'outputs.txt'
file_covid = 'outputs.txt'

# Caminhos completos dos arquivos
path_normal = os.path.join(dir_normal, file_normal)
path_covid = os.path.join(dir_covid, file_covid)

nc = 30
ns = 1000
k = (nc-1)*ns

# Leitura dos dados
t, Vsa, Vsv, Vpa, Vpv, Vra, Vla, Vrv, Vlv, psa, psv, ppa, ppv, pra, pla, prv, plv, Era, Ela, Erv, Elv = np.loadtxt(path_normal, skiprows=1, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], unpack=True)
t_CV, Vsa_CV, Vsv_CV, Vpa_CV, Vpv_CV, Vra_CV, Vla_CV, Vrv_CV, Vlv_CV, psa_CV, psv_CV, ppa_CV, ppv_CV, pra_CV, pla_CV, prv_CV, plv_CV, Era_CV, Ela_CV, Erv_CV, Elv_CV = np.loadtxt(path_covid, skiprows=1, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], unpack=True)

# #Gráficos Comparativos 
# plt.plot(t[k:],Era[k:], color = 'purple', label ='Era')
# plt.plot(t[k:],Era_CV[k:], color = 'black', label ='Era_CV')
# plt.legend(loc='best')
# plt.xlabel('Time [s]')
# plt.ylabel('Elastance')
# plt.show()

# plt.plot(t[k:],Ela[k:], color = 'red', label ='Ela')
# plt.plot(t[k:],Ela_CV[k:], color = 'green', label ='Ela_CV')
# plt.legend(loc='best')
# plt.xlabel('Time [s]')
# plt.ylabel('Elastance')
# plt.show()

# plt.plot(t[k:],Erv[k:], color = 'blue', label ='Erv')
# plt.plot(t[k:],Erv_CV[k:], color = 'orange', label ='Erv_CV')
# plt.legend(loc='best')
# plt.xlabel('Time [s]')
# plt.ylabel('Elastance')
# plt.show()

# plt.plot(t[k:],Elv[k:], color = 'grey', label ='Elv')
# plt.plot(t[k:],Elv_CV[k:], color = 'pink', label ='Elv_CV')
# plt.legend(loc='best')
# plt.xlabel('Time [s]')
# plt.ylabel('Elastance')
# plt.show()

#plt.plot(t[k:],psa[k:], color='blue', label='normal')
plt.plot(t_CV[k:], psa_CV[k:], color='red', label='covid')
plt.legend(loc='best')
plt.xlabel('volume [mL]')
plt.ylabel('pressão [mmHg]')
plt.tight_layout()
plt.savefig('fig_psa_covid.png', format='png', dpi=300)
plt.savefig('fig_psa_covid.pdf', format='pdf')
plt.show()

plt.plot(Vrv[k:], prv[k:], color='blue', label='normal')
plt.plot(Vrv_CV[k:], prv_CV[k:], color='red', label='covid')
#plt.legend(loc='best')
plt.xlabel('volume [mL]')
plt.ylabel('pressão [mmHg]')
plt.title('ventrículo direito (RV)')
plt.tight_layout()
plt.savefig('fig_pvloop_rv.png', format='png', dpi=300)
plt.savefig('fig_pvloop_rv.pdf', format='pdf')
plt.show()

plt.plot(Vlv[k:], plv[k:], color='blue', label='normal')
plt.plot(Vlv_CV[k:], plv_CV[k:], color='red', label='covid')
plt.legend(loc='best')
plt.xlabel('Vlv [mL]')
plt.ylabel('Plv[mmHg]')
plt.xlim([38,125])
plt.title('ventrículo esquerdo (LV)')
plt.tight_layout()
plt.savefig('fig_pvloop_lv.png', format='png', dpi=300)
plt.savefig('fig_pvloop_lv.pdf', format='pdf')
plt.show()

# Criando subplots lado a lado
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Primeiro subplot
axes[0].plot(Vlv[k:], plv[k:], color='tab:blue', label='normal')
axes[0].plot(Vlv_CV[k:], plv_CV[k:], color='red', label='covid')
axes[0].legend(loc='best')
axes[0].set_xlabel('Vlv [mL]')
axes[0].set_ylabel('Plv [mmHg]')

# Segundo subplot
axes[1].plot(Vrv[k:], prv[k:], color='tab:blue', label='normal')
axes[1].plot(Vrv_CV[k:], prv_CV[k:], color='red', label='covid')
axes[1].legend(loc='best')
axes[1].set_xlabel('Vrv [mL]')
axes[1].set_ylabel('Prv [mmHg]')

# Ajustando o layout para evitar sobreposição
plt.tight_layout()

#Salvando gráficos
plt.savefig('pvloop_juntos.pdf', format='pdf', dpi=300)
# Exibindo os gráficos
plt.show()

#Graficos para pressao volume 
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Espaçamento entre os subplots
plt.subplots_adjust(hspace=0.5, wspace=0.4)

# Primeiro subplot
axes[0,0].plot(t[k:], plv[k:], color='tab:blue')
axes[0,0].set_xlabel('tempo [s]')
axes[0,0].set_title(r'$P_{lv}$')
axes[0,0].set_ylabel('pressão [mmHg]')

# Segundo subplot
axes[0,1].plot(t[k:], Vlv[k:], color='tab:blue')
axes[0,1].set_xlabel('t [s]')
axes[0,1].set_title(r'$V_{lv}$')
axes[0,1].set_ylabel('volume [mL]')

#terceiro subplot
axes[1,0].plot(t[k:], prv[k:], color='tab:blue')
axes[1,0].set_xlabel('tempo [s]')
axes[1,0].set_title(r'$P_{rv}$')
axes[1,0].set_ylabel('pressão [mmHg]')

#quarto subplot
axes[1,1].plot(t[k:], Vrv[k:], color='tab:blue')
axes[1,1].set_xlabel('t [s]')
axes[1,1].set_title(r'$V_{rv}$')
axes[1,1].set_ylabel('volume [mL]')

# Ajustando o layout para evitar sobreposição
plt.tight_layout()

#Salvando gráficos
plt.savefig('subplot_ventriculos_NORMAL.pdf', format='pdf', dpi=300)
# Exibindo os gráficos
plt.show()

#Graficos para pressao volume 
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Espaçamento entre os subplots
plt.subplots_adjust(hspace=0.5, wspace=0.4)

# Primeiro subplot
axes[0,0].plot(t_CV[k:], plv_CV[k:], color='red')
axes[0,0].set_xlabel('tempo [s]')
axes[0,0].set_title(r'$P_{lv}$')
axes[0,0].set_ylabel('pressão [mmHg]')

# Segundo subplot
axes[0,1].plot(t_CV[k:], Vlv_CV[k:], color='red')
axes[0,1].set_xlabel('tempo [s]')
axes[0,1].set_title(r'$V_{lv}$')
axes[0,1].set_ylabel('volume [mL]')

#terceiro subplot
axes[1,0].plot(t_CV[k:], prv_CV[k:], color='red')
axes[1,0].set_xlabel('tempo [s]')
axes[1,0].set_title(r'$P_{rv}$')
axes[1,0].set_ylabel('pressão [mmHg]')

#quarto subplot
axes[1,1].plot(t_CV[k:], Vrv_CV[k:], color='red')
axes[1,1].set_xlabel('tempo [s]')
axes[1,1].set_title(r'$V_{rv}$')
axes[1,1].set_ylabel('volume [mL]')

# Ajustando o layout para evitar sobreposição
plt.tight_layout()

#Salvando gráficos
plt.savefig('subplot_ventriculos_COVID.pdf', format='pdf', dpi=300)
# Exibindo os gráficos
plt.show()