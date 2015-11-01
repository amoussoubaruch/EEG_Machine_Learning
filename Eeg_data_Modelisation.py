# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:41:54 2015

@author: amoussoubaruch
"""
############################################################################
# EEG Machine Learning Challenge
# Supervised Learning
############################################################################
#####Import librarie
###%matplotlib inline
import mlpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##import mne
from scipy import io
import mlpy.wavelet as wave
help(mlpy)
#####Importation des fonction de pyeeg
from scipy import stats
from pyeeg import hurst
from pyeeg import pfd
from pyeeg import hfd
from pyeeg import hjorth
from pyeeg import spectral_entropy
from pyeeg import svd_entropy
from pyeeg import fisher_info
from pyeeg import  ap_entropy
from pyeeg import samp_entropy
from pyeeg import dfa
from pyeeg import bin_power
from scipy import fftpack
import time

'''Importation de la base de donnée'''
chemin='/cal/homes/amoussou/Downloads/data_challenge.mat'
dataset = io.loadmat(chemin)
type(dataset)

####Affichage de tous les éléments du dico
dataset.items() 
len(dataset)
dataset['X_train'].shape
dataset['X_train'][0].shape

###Taille du vercteur des Y
dataset['y_train'].shape


###Récupération des bases x y et test pour les alogo

X_train, y_train, X_test = dataset['X_train'], dataset['y_train'], dataset['X_test']



'''Correction de base sur les siganux pour les moyennés à zéro'''

####Train
###Moyenne par ligne
Xm=X_train.mean(axis=1)
Xm.shape
###Centrage des valeurs du signal autour de zéro
X_train = X_train - Xm[:, np.newaxis]

X_train.shape
####Test
Xm1=X_test.mean(axis=1)
Xm1.shape
X_test = X_test - Xm1[:, np.newaxis]

X_test.shape

''' TRANSFORME DE FOUIRIER ET NOUVELLES FEATURES'''

''' Tranformé de fourrier'''  ###Okay


###on divise par la taille du vecteur pour normaliser la fft


# fréquence d'échantillonage
'''Nous avons supposer que nous avons 6000 points N pour uen durée de 30
secondes, ce qui signifie que le fréquence d'échantillonage est 200'''

FreqEch = 200                 
PerEch = 1./FreqEch             # période d'échantillonnage
N = FreqEch*(30 - 0)


'''Représentation du specte du premier signal'''
signal=dataset['X_train'][1]
t = np.linspace(0, 30, N)
# définition des données de FFT
FenAcq = signal.size             # taille de la fenetre temporelle
    
# calcul de la TFD par l'algo de FFT
signal_FFT = abs(fftpack.fft(signal))    # on ne récupère que les composantes réelles
###passebande=passe_bas(signal_FFT,0)
# récupération du domaine fréquentiel
signal_freq = fftpack.fftfreq(FenAcq,PerEch)

#affichage du signal
plt.subplot(211)
plt.title('Signal et son spectre')
##plt.ylim()
plt.plot(t, signal)
plt.xlabel('Temps (s)'); plt.ylabel('Amplitude')

#affichage du spectre du signal
plt.subplot(212)
plt.xlim()
plt.plot(signal_freq,signal_FFT)
plt.xlabel('Frequence (Hz)'); plt.ylabel('Amplitude')
#plt.title('Signal et son spectre')
plt.show()




## extraction des valeurs réelles de la FFT et du domaine fréquentiel
#signal_FFT = signal_FFT[0:len(signal_FFT)//2]
#signal_freq = signal_freq[0:len(signal_freq)//2]
#max=0
#maxIndex= 0
#for i in range(len(signal_FFT)):
#    if signal_FFT[i] > max:
#        max = signal_FFT[i]
#        maxIndex = i        
#freq_fond = float(maxIndex*200.00) /(N/2)


########## Récupération des valeurs obtenus par transforme de Fourier pour chaque signal
fft_features_train=[]
for i in range(X_train.shape[0]):
    ##print i
###Nous divisions par la taille de X_train pour normaliser la fft
    h=abs(fftpack.fft(X_train[i,],1200)/(X_train[i,]).size)
    h1=h[0:len(h)//2]
    fft_features_train.append(h1)


fft_features_test=[]
for i in range(X_test.shape[0]):
    ##print i
    h=abs(fftpack.fft(X_test[i,],1200)/(X_test[i,]).size)
    h1=h[0:len(h)//2]
    fft_features_test.append(h1)

##########Nous souhaitons récupérer la fréquence fondammentale de chauq signale

fftmax_features_train=[]
for i in range(X_train.shape[0]):
    ##print i
    h=abs(fftpack.fft(X_train[i,]))
    hfft=h[0:len(h)//2]
    max=0
    maxIndex= 0
    for i in range(len(hfft)):
        if hfft[i] > max:
            max = hfft[i]
            maxIndex = i        
    freq_fond = signal_freq[maxIndex]
    fftmax_features_train.append(freq_fond)


fftmax_features_test=[]
for i in range(X_test.shape[0]):
    ##print i
    h=abs(fftpack.fft(X_test[i,]))
    hfft=h[0:len(h)//2]
    max=0
    maxIndex= 0
    for i in range(len(hfft)):
         if hfft[i] > max:
             max = hfft[i]
             maxIndex = i 
    freq_fond = signal_freq[maxIndex]
    fftmax_features_test.append(freq_fond)




########### Mise en place de nouvelles variables

    
'''Petrosian Fractal Dimension (PFD)''' ###Okay
pfd_features_train=[]
for i in range(X_train.shape[0]):
    ##print i
    h=pfd(X_train[i,])
    pfd_features_train.append(h)


pfd_features_test=[]
for i in range(X_test.shape[0]):
    ##print i
    h=pfd(X_test[i,])
    pfd_features_test.append(h)

'''Higuchi Fractal Dimension (HFD) ''' ##Okay
hfd_features_train=[]
for i in range(X_train.shape[0]):
    ##print i
    h=hfd(X_train[i,],5)
    hfd_features_train.append(h)


hfd_features_test=[]
for i in range(X_test.shape[0]):
    ##print i
    h=hfd(X_test[i,],5)
    hfd_features_test.append(h)

'''Hjorth Parameters  ''' ###Okay
hjorth_features_train=[]
for i in range(X_train.shape[0]):
    ##print i
    h=hjorth(X_train[i,])
    hjorth_features_train.append(h)


hjorth_features_test=[]
for i in range(X_test.shape[0]):
    ##print i
    h=hjorth(X_test[i,])
    hjorth_features_test.append(h)

'''Spectral Entropy  '''  ##Okay
spectral_entropy_features_train=[]
for i in range(X_train.shape[0]):
    ##print i
    h=spectral_entropy(X_train[i,],[0.54,5,7,12,50],173,Power_Ratio = None)
    spectral_entropy_features_train.append(h)


spectral_entropy_features_test=[]
for i in range(X_test.shape[0]):
    ##print i
    h=spectral_entropy(X_test[i,],[0.54,5,7,12,50],173,Power_Ratio = None)
    spectral_entropy_features_test.append(h)
    
'''SVD Entropy  ''' ## Okay
svd_entropy_features_train=[]
for i in range(X_train.shape[0]):
    ##print i
    h=svd_entropy(X_train[i,],4,10, W = None)
    svd_entropy_features_train.append(h)


svd_entropy_features_test=[]
for i in range(X_test.shape[0]):
    ##print i
    h=svd_entropy(X_test[i,],4,10, W = None)
    svd_entropy_features_test.append(h)
    

'''Fisher Information  '''##Okay
fisher_info_features_train=[]
for i in range(X_train.shape[0]):
    ##print i
    h=fisher_info(X_train[i,],4,10, W = None)
    fisher_info_features_train.append(h)


fisher_info_features_test=[]
for i in range(X_test.shape[0]):
    ##print i
    h=fisher_info(X_test[i,],4,10, W = None)
    fisher_info_features_test.append(h)
    
    
''' Detrended Fluctuation Analysis'''  ###Okay
dfa_features_train=[]
for i in range(X_train.shape[0]):
    ##print i
    h=dfa(X_train[i,],Ave = None, L = None)
    dfa_features_train.append(h)


dfa_features_test=[]
for i in range(X_test.shape[0]):
    ##print i
    h=dfa(X_test[i,],Ave = None, L = None)
    dfa_features_test.append(h)

#''' bin_power(), Power Spectral Density (PSD), spectrum power in a set of frequency bins, and, Relative Intensity Ratio (RIR)'''
#bin_power_features_train=[]  ###Okay
#for i in range(X_train.shape[0]):
#    ##print i
#    h=bin_power(X_train[i,],[0.54,5,7,12,50],173)
#    bin_power_features_train.append(h)
#
##Power_features_train = bin_power_features_train[]
##Power_Ratio_features_train	= bin_power_features_train[]
#
#bin_power_features_test=[]
#for i in range(X_test.shape[0]):
#    ##print i
#    h=bin_power(X_test[i,],[0.54,5,7,12,50],173)
#    bin_power_features_test.append(h)
#
###########Ratio
#bin_power_features_train1=[]  ###Okay
#for i in range(X_train.shape[0]):
#    ##print i
#    h=bin_power(X_train[i,],[0.54,5,7,12,50],173)
#    bin_power_features_train.append(h)
#
##Power_features_train = bin_power_features_train[]
##Power_Ratio_features_train	= bin_power_features_train[]
#
#bin_power_features_test1=[]
#for i in range(X_test.shape[0]):
#    ##print i
#    h=bin_power(X_test[i,],[0.54,5,7,12,50],173)
#    bin_power_features_test.append(h)



####### Constitution de notre vecteur de feaure finale
XX_train = np.c_[fftmax_features_train,fft_features_train,dfa_features_train,fisher_info_features_train,svd_entropy_features_train,spectral_entropy_features_train,hjorth_features_train,hfd_features_train,pfd_features_train,np.mean(X_train, axis=1), np.std(X_train, axis=1), stats.kurtosis(X_train, axis=1),stats.skew(X_train, axis=1),np.max(X_train, axis=1)-np.min(X_train, axis=1)]
XX_train.head()
XX_test = np.c_[fftmax_features_test,fft_features_test,dfa_features_test,fisher_info_features_test,svd_entropy_features_test,spectral_entropy_features_test,hjorth_features_test,hfd_features_test,pfd_features_test,np.mean(X_test, axis=1), np.std(X_test, axis=1), stats.kurtosis(X_test, axis=1),stats.skew(X_test, axis=1),np.max(X_test, axis=1)-np.min(X_test, axis=1)]


from sklearn.ensemble import RandomForestClassifier
clf_rd = RandomForestClassifier(n_estimators=20, max_depth=50,min_samples_split=30)
clf=clf_rd.fit(XX_train, y_train)
y_pred=clf_rd.predict(XX_test)


from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=10))
clf=clf.fit(XX_train, y_train)
y_pred=clf_rd.predict(XX_test)



XX_train= np.c_[fftmax_features_train,fft_features_train,np.mean(X_train, axis=1), np.std(X_train, axis=1), stats.kurtosis(X_train, axis=1),stats.skew(X_train, axis=1),np.max(X_train, axis=1),np.min(X_train, axis=1)]

XX_test = np.c_[fftmax_features_test,fft_features_test,np.mean(X_test, axis=1), np.std(X_test, axis=1), stats.kurtosis(X_test, axis=1),stats.skew(X_test, axis=1),np.max(X_test, axis=1),np.min(X_test, axis=1)]

XX_train.shape
XX_test.shape
###### Notre échantillon d'apprentissage XX_train, nous le transformaons en apprentissage
###### et test. le X_test de départ nous servira de validation

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation


xxx_train, xxx_test, yyy_train, y_test = train_test_split(XX_train, y_train, test_size=0.30)

###Modèle 1 : Arbre de décision
from sklearn import tree
profondeur_maxi=range(10,100)
dicoarbre={}
for a in profondeur_maxi:
    clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=a).fit(xxx_train, yyy_train)
    scores=clf.score(xxx_test, y_test)   
    dicoarbre[a]=scores
table_arbre=pd.DataFrame(dicoarbre.items(), columns=['profondeur', 'score']) 

Max_prof=table_arbre[table_arbre.score == table_arbre['score'].max() ]


##Arbre de décison avec profondeur mawimale
clfmax = tree.DecisionTreeClassifier(criterion='entropy',max_depth=Max_prof).fit(xxx_train, yyy_train)
y_pred=clfmax.predict(XX_test)

np.savetxt('/cal/homes/amoussou/Downloads/y_pred.txt', y_pred, fmt='%s')

####Score challenge en ligne : 

#####Random forest
from sklearn.ensemble import RandomForestClassifier

estimator=range(10,100)
dico_rd={}
for i in estimator:
    clf_rd = RandomForestClassifier(n_estimators=i, max_depth=10,min_samples_split=1).fit(xxx_train, yyy_train)
    scores = clf.score(xxx_test, y_test) 
    scores.mean()   
    dico_rd[a]=scores

table_rd=pd.DataFrame(dico_rd.items(), columns=['Classifieur', 'score']) 

Max_prof=table_rd[table_rd.score == table_rd['score'].max() ]

##    Random Forest optimal
clf_rd = RandomForestClassifier(n_estimators=20, max_depth=10,min_samples_split=1).fit(xxx_train, yyy_train)
y_pred=clf_rd.predict(XX_test)

np.savetxt('/cal/homes/amoussou/Downloads/y_predr.txt', y_pred, fmt='%s')

####aDBOOSTRAP
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=10))
clf=clf.fit(XX_train, y_train)

y_predr=clf.predict(XX_test)

###Sauvegarde des prédictions
np.savetxt('/cal/homes/amoussou/Downloads/y_predr.txt', y_predr, fmt='%s')



