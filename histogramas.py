import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import statistics
plt.style.use('nature')


#Função de calcula média, moda, mediana e desvio padrão das QoI's
def estatisticas(x):
    media = np.mean(x)
    moda= statistics.mode(x)
    mediana= np.median(x)
    desvio= np.std(x)
    return media,moda,mediana,desvio

if __name__ == "__main__":

    arq = 'QOIs/matriz_QOI.txt'
    Psa_min,Psa_max,Psa_mean,PAP_mean,PPV_mean,LV_EDV,LV_ESV,LV_SV,LV_EF,LV_Pmax,RV_EDV,RV_ESV,RV_SV,RV_EF,RV_Pmax = np.loadtxt(arq, skiprows = 1, usecols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],unpack = True)

    plt.subplot(2,2,1)
    plt.hist(Psa_min, bins=30, color='darkgreen',density=True)
    plt.title('A histogram of Psa_min')
    plt.subplot(2,2,2)
    plt.hist(Psa_max, bins=30,color='darkgreen',density=True)
    plt.title('A histogram of Psa_max')
    plt.subplot(2,2,3)
    plt.hist(Psa_mean, bins=30,color='red',density=True)
    plt.title('A histogram of Psa_mean')
    plt.subplot(2,2,4)
    plt.hist(PAP_mean, bins=30,color='red',density=True)
    plt.title('A histogram of PAP_mean')
    plt.show()

    plt.subplot(2,2,1)
    plt.hist(PPV_mean, bins=30,color='orange',density=True)
    plt.title('A histogram of PPV_mean')
    plt.subplot(2,2,2)
    plt.hist(LV_EDV, bins=30,color='orange',density=True)
    plt.title('A histogram of LV_EDV')
    plt.subplot(2,2,3)
    plt.hist(LV_ESV, bins=30,color='blue',density=True)
    plt.title('A histogram of LV_ESV')
    plt.subplot(2,2,4)
    plt.hist(LV_SV, bins=30,color='blue',density=True)
    plt.title('A histogram of LV_SA')
    plt.show()

    plt.subplot(2,2,1)
    plt.hist(LV_EF, bins=30,color='purple',density=True)
    plt.title('A histogram of LV_EF')
    plt.subplot(2,2,2)
    plt.hist(LV_Pmax, bins=30,color='purple',density=True)
    plt.title('A histogram of LV_Pmax')
    plt.subplot(2,2,3)
    plt.hist(RV_EDV, bins=30,color='grey',density=True)
    plt.title('A histogram of RV_EDV')
    plt.subplot(2,2,4)
    plt.hist(RV_ESV, bins=30,color='grey',density=True)
    plt.title('A histogram of RV_ESV')
    plt.show()

    plt.subplot(2,2,1)
    plt.hist(RV_SV, bins=30,color='pink',density=True)
    plt.title('A histogram of RV_SV')
    plt.subplot(2,2,2)
    plt.hist(RV_EF, bins=30,color='pink',density=True)
    plt.title('A histogram of RV_EF')
    plt.subplot(2,2,3)
    plt.hist(RV_Pmax, bins=30,color='pink',density=True)
    plt.title('A histogram of RV_Pmax')
    plt.show()

    #Cálculo das quantidades estatísticas 
    media_Psa_min, moda_Psa_min, mediana_Psa_min, dp_Psa_min = estatisticas(Psa_min)
    print("Psa_min ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_Psa_min, moda_Psa_min, mediana_Psa_min, dp_Psa_min))

    media_Psa_max, moda_Psa_max, mediana_Psa_max, dp_Psa_max = estatisticas(Psa_max)
    print("Psa_max ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_Psa_max, moda_Psa_max, mediana_Psa_max, dp_Psa_max))

    media_Psa_mean, moda_Psa_mean, mediana_Psa_mean, dp_Psa_mean = estatisticas(Psa_mean)
    print("Psa_mean ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_Psa_mean, moda_Psa_mean, mediana_Psa_mean, dp_Psa_mean))

    media_PAP_mean, moda_PAP_mean, mediana_PAP_mean, dp_PAP_mean = estatisticas(PAP_mean)
    print("PAP_mean ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_PAP_mean, moda_PAP_mean, mediana_PAP_mean, dp_PAP_mean))

    media_PPV_mean, moda_PPV_mean, mediana_PPV_mean, dp_PPV_mean = estatisticas(PPV_mean)
    print("PPV_mean ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_PPV_mean, moda_PPV_mean, mediana_PPV_mean, dp_PPV_mean))

    media_LV_EDV, moda_LV_EDV, mediana_LV_EDV, dp_LV_EDV = estatisticas(LV_EDV)
    print("LV_EDV ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_LV_EDV, moda_LV_EDV, mediana_LV_EDV, dp_LV_EDV))

    media_LV_ESV, moda_LV_ESV, mediana_LV_ESV, dp_LV_ESV = estatisticas(LV_ESV)
    print("LV_ESV ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_LV_ESV, moda_LV_ESV, mediana_LV_ESV, dp_LV_ESV))

    media_LV_SV, moda_LV_SV, mediana_LV_SV, dp_LV_SV = estatisticas(LV_SV)
    print("LV_SV ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_LV_SV, moda_LV_SV, mediana_LV_SV, dp_LV_SV))

    media_LV_EF, moda_LV_EF, mediana_LV_EF, dp_LV_EF = estatisticas(LV_EF)
    print("LV_EF ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_LV_EF, moda_LV_EF, mediana_LV_EF, dp_LV_EF))

    media_LV_Pmax, moda_LV_Pmax, mediana_LV_Pmax, dp_LV_Pmax = estatisticas(LV_Pmax)
    print("LV_Pmax ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_LV_Pmax, moda_LV_Pmax, mediana_LV_Pmax, dp_LV_Pmax))

    media_RV_EDV, moda_RV_EDV, mediana_RV_EDV, dp_RV_EDV = estatisticas(RV_EDV)
    print("RV_EDV ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_RV_EDV, moda_RV_EDV, mediana_RV_EDV, dp_RV_EDV))

    media_RV_ESV, moda_RV_ESV, mediana_RV_ESV, dp_RV_ESV = estatisticas(RV_ESV)
    print("RV_ESV ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_RV_ESV, moda_RV_ESV, mediana_RV_ESV, dp_RV_ESV))

    media_RV_SV, moda_RV_SV, mediana_RV_SV, dp_RV_SV = estatisticas(RV_SV)
    print("RV_SV ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_RV_SV, moda_RV_SV, mediana_RV_SV, dp_RV_SV))

    media_RV_EF, moda_RV_EF, mediana_RV_EF, dp_RV_EF = estatisticas(RV_EF)
    print("RV_EF ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_RV_EF, moda_RV_EF, mediana_RV_EF, dp_RV_EF))

    media_RV_Pmax, moda_RV_Pmax, mediana_RV_Pmax, dp_RV_Pmax = estatisticas(RV_Pmax)
    print("RV_Pmax ->> Média = {0}, moda = {1}, mediana = {2}, desvio-padrão = {3}".format(media_RV_Pmax, moda_RV_Pmax, mediana_RV_Pmax, dp_RV_Pmax))
