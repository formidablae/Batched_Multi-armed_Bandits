import math
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from prepare_data import prepare_data
from UCB1_data import UCB1_data
from BASEFunc_data import BASEFunc_data
from PRCS_twoarm_data import PRCS_twoarm_data

# Read dataset from file
print('Inizio lettura dataset film...')
movies = pd.read_csv("../../Datasets/ml-10M100K/movies.dat", sep='[:][:]', engine='python', header=None)
print('Fine lettura dataset film.')
print('Inizio lettura dataset recensioni...')
ratings = pd.read_csv("../../Datasets/ml-10M100K/ratings.dat", sep='[:][:]', engine='python', header=None)
print('Fine lettura dataset recensioni.')

"""
Parametri
"""

# K = 3 # Number of genres (arms) that the chosen user (bandit) has rated the most
# M = 3  # Numero dei batches
# m = 100
K = 3
M = 3
m = 200  # Users/Bandits = 200

# Prepare data
ratingsOfChosenBanditWithGenres = prepare_data(movies, ratings, m)

# K_set = vettore riga di 6 punti da log10(2) a log10(18) ugualmente
# logaritmicamente distanti (arrotondati). 18 e non 120 perche gli arms
# del dataset sono 18
# M_set = insieme dei punti da 2 a 7 (compreso).
K_set = np.round(np.logspace(math.log(2, 10),
                             math.log(18, 10),
                             num=6,
                             endpoint=True,
                             base=10.0,
                             dtype=None, axis=0))
M_set = range(2, 8)

# mu_max = 2 reward medio per l'arm ottimale
# gamma = 1
mu_max = 2  # Max rating/reward of movies/arm pull = 1
mu_min = 1  # Min rating/reward of movies/arm pull = -1
gamma = 1

"""
Esperimenti
"""

# dipendenza da M

# regretMinimax_M = matrice di zeri di dimensione mxlen(M_set)
# regretGeometric_M = matrice di zeri di dimensione mxlen(M_set)
# regretArithmetic_M = matrice di zeri di dimensione mxlen(M_set)
# regretUCB_M = vettore di zeri di dimensione mx1
# mu = i rewards come concatenazione (bind di colonne) dei
# reward di mu_max e tutti i K-1 mu_min
regretMinimax_M = np.zeros((m, len(M_set)), dtype=float)
regretGeometric_M = np.zeros((m, len(M_set)), dtype=float)
regretArithmetic_M = np.zeros((m, len(M_set)), dtype=float)
regretUCB_M = np.zeros((m,), dtype=float)
mu = np.concatenate(([mu_max], mu_min * np.ones((1, K-1),
                                                dtype=int).flatten()), axis=0)

# T = numero di pulls nel time orizon
T = 0
T_allVariations = []


for iter_i in range(0, m):
    print("Calcolando Diagramma 1, iter_i(da 0 a ", str(m - 1), ") =", str(iter_i)) 

    # Choosing the only the ratings of movies from the ith bandit
    ratingsOfChosenBanditWithGenresith = \
        ratingsOfChosenBanditWithGenres[ratingsOfChosenBanditWithGenres[
            'Bandit'].isin(ratingsOfChosenBanditWithGenres['Bandit'].value_counts()[iter_i:iter_i+1].index.tolist())]

    leastLongTimeHorizon = 0
    ratingsOfChosenBanditWithGenresithdata = ratingsOfChosenBanditWithGenresith

    while leastLongTimeHorizon < 2:
        # Choosing the ratings on the k arms/genres most rated from the ith bandit
        ratingsOfChosenBanditWithGenresithdata = ratingsOfChosenBanditWithGenresith[
            ratingsOfChosenBanditWithGenresith[
                'Arm_K'].isin(ratingsOfChosenBanditWithGenresith['Arm_K'].value_counts()[:K].index.tolist())]

        leastLongTimeHorizon = ratingsOfChosenBanditWithGenresithdata['Arm_K'].value_counts().min()

        # Dropping di rating rows per genre/arm oltre il leastLongTimeHorizon
        # Renumbering of the arms from 1 to K
        # Numbering time horizin
        newData = []
        k = 1
        for arm in ratingsOfChosenBanditWithGenresithdata['Arm_K'].unique().tolist():
            newDataFrame = ratingsOfChosenBanditWithGenresithdata[
                ratingsOfChosenBanditWithGenresithdata['Arm_K'] == arm].head(leastLongTimeHorizon)
            newDataFrame['Arm_K'] = k
            k = k + 1
            newDataFrame['Time_T'] = range(1, leastLongTimeHorizon + 1)
            newData.append(newDataFrame)

        ratingsOfChosenBanditWithGenresithdata = pd.concat(newData)

        if len(ratingsOfChosenBanditWithGenresithdata['Arm_K'].unique().tolist()) < K:
            K = K - 1
        else:
            break

    T = leastLongTimeHorizon
    T_allVariations.append(T)

    regretUCB_M[iter_i] = UCB1_data(ratingsOfChosenBanditWithGenresithdata, mu, K, T)
    for iter_M in range(0, len(M_set)):
        temp_M = M_set[iter_M]
        regretMinimax_M[iter_i, iter_M] = \
            BASEFunc_data(ratingsOfChosenBanditWithGenresithdata, mu, K, T, temp_M, 'minimax', gamma)
        regretGeometric_M[iter_i, iter_M] = \
            BASEFunc_data(ratingsOfChosenBanditWithGenresithdata, mu, K, T, temp_M, 'geometric', gamma)
        regretArithmetic_M[iter_i, iter_M] = \
            BASEFunc_data(ratingsOfChosenBanditWithGenresithdata, mu, K, T, temp_M, 'arithmetic', gamma)

T_first_min = min(T_allVariations)
T_first_max = max(T_allVariations)
T = np.mean(T_allVariations, axis=0)
T_first = T

regretMinimax_M_mean = np.mean(regretMinimax_M, axis=0) / T
regretGeometric_M_mean = np.mean(regretGeometric_M, axis=0) / T
regretArithmetic_M_mean = np.mean(regretArithmetic_M, axis=0) / T
regretUCB_M_mean = np.mean(regretUCB_M, axis=0) / T

# dependence on K

regretMinimax_K = np.zeros((m, len(K_set)), dtype=float)
regretGeometric_K = np.zeros((m, len(K_set)), dtype=float)
regretArithmetic_K = np.zeros((m, len(K_set)), dtype=float)
regretUCB_K = np.zeros((m, len(K_set)), dtype=float)

T_allVariations.clear()
for iter_i in range(0, m):
    print("Calcolando Diagramma 2, iter_i(da 0 a ", str(m - 1), ") =", str(iter_i)) 

    # Choosing the only the ratings of movies from the ith bandit
    ratingsOfChosenBanditWithGenresith = \
        ratingsOfChosenBanditWithGenres[ratingsOfChosenBanditWithGenres[
            'Bandit'].isin(ratingsOfChosenBanditWithGenres['Bandit'].value_counts()[iter_i:iter_i+1].index.tolist())]

    for iter_K in range(0, len(K_set)):
        temp_K = int(K_set[iter_K])
        leastLongTimeHorizon = 0
        while 1:
            # Choosing the ratings on the k arms/genres most rated from the ith bandit
            ratingsOfChosenBanditWithGenresithdata = ratingsOfChosenBanditWithGenresith[
                ratingsOfChosenBanditWithGenresith[
                    'Arm_K'].isin(ratingsOfChosenBanditWithGenresith['Arm_K'].value_counts()[:temp_K].index.tolist())]

            leastLongTimeHorizon = ratingsOfChosenBanditWithGenresithdata['Arm_K'].value_counts().min()

            # Dropping di rating rows per genre/arm oltre il leastLongTimeHorizon
            # Renumbering of the arms from 1 to K
            # Numbering time horizin
            newData = []
            k = 1
            for arm in ratingsOfChosenBanditWithGenresithdata['Arm_K'].unique().tolist():
                newDataFrame = ratingsOfChosenBanditWithGenresithdata[
                    ratingsOfChosenBanditWithGenresithdata['Arm_K'] == arm].head(leastLongTimeHorizon)
                newDataFrame['Arm_K'] = k
                k = k + 1
                newDataFrame['Time_T'] = range(1, leastLongTimeHorizon + 1)
                newData.append(newDataFrame)

            ratingsOfChosenBanditWithGenresithdata = pd.concat(newData)

            if len(ratingsOfChosenBanditWithGenresithdata['Arm_K'].unique().tolist()) < temp_K:
                temp_K = temp_K - 1
            else:
                break

        T = leastLongTimeHorizon
        T_allVariations.append(T)

        mu = np.concatenate(([mu_max], mu_min *
                             np.ones((1, temp_K - 1),
                                     dtype=int).flatten()), axis=0)

        regretUCB_K[iter_i, iter_K] = UCB1_data(ratingsOfChosenBanditWithGenresithdata, mu, temp_K, T)
        regretMinimax_K[iter_i, iter_K] = \
            BASEFunc_data(ratingsOfChosenBanditWithGenresithdata, mu, temp_K, T, M, 'minimax', gamma)
        regretGeometric_K[iter_i, iter_K] = \
            BASEFunc_data(ratingsOfChosenBanditWithGenresithdata, mu, temp_K, T, M, 'geometric', gamma)
        regretArithmetic_K[iter_i, iter_K] = \
            BASEFunc_data(ratingsOfChosenBanditWithGenresithdata, mu, temp_K, T, M, 'arithmetic', gamma)

T = np.mean(T_allVariations, axis=0)

regretMinimax_K_mean = np.mean(regretMinimax_K, axis=0) / T
regretGeometric_K_mean = np.mean(regretGeometric_K, axis=0) / T
regretArithmetic_K_mean = np.mean(regretArithmetic_K, axis=0) / T
regretUCB_K_mean = np.mean(regretUCB_K, axis=0) / T

# dependence on T

K = 3
T = T_first

# logaritmicamente distanti (arrotondati)
T_set = np.round(np.logspace(math.log(T_first_min, 10), math.log(T_first_max, 10), num=6,
                             endpoint=True,
                             base=10.0,
                             dtype=None, axis=0))

regretMinimax_T = np.zeros((m, len(T_set)), dtype=float)
regretGeometric_T = np.zeros((m, len(T_set)), dtype=float)
regretArithmetic_T = np.zeros((m, len(T_set)), dtype=float)
regretUCB_T = np.zeros((m, len(T_set)), dtype=float)
mu = np.concatenate(([mu_max], mu_min * np.ones((1, K - 1),
                                                dtype=int).flatten()), axis=0)

for iter_i in range(0, m):
    print("Calcolando Diagramma 3, iter_i(da 0 a ", str(m - 1), ") =", str(iter_i)) 

    # Choosing the only the ratings of movies from the ith bandit
    ratingsOfChosenBanditWithGenresith = \
        ratingsOfChosenBanditWithGenres[ratingsOfChosenBanditWithGenres[
            'Bandit'].isin(ratingsOfChosenBanditWithGenres['Bandit'].value_counts()[iter_i:iter_i + 1].index.tolist())]

    for iter_T in range(0, len(T_set)):
        leastLongTimeHorizon = 0
        ratingsOfChosenBanditWithGenresithdata = ratingsOfChosenBanditWithGenresith

        while leastLongTimeHorizon < 2:
            # Choosing the ratings on the k arms/genres most rated from the ith bandit
            ratingsOfChosenBanditWithGenresithdata = ratingsOfChosenBanditWithGenresith[
                ratingsOfChosenBanditWithGenresith[
                    'Arm_K'].isin(ratingsOfChosenBanditWithGenresith['Arm_K'].value_counts()[:K].index.tolist())]

            leastLongTimeHorizon = ratingsOfChosenBanditWithGenresithdata['Arm_K'].value_counts().min()

            # Dropping di rating rows per genre/arm oltre il leastLongTimeHorizon
            # Renumbering of the arms from 1 to K
            # Numbering time horizin
            newData = []
            k = 1
            for arm in ratingsOfChosenBanditWithGenresithdata['Arm_K'].unique().tolist():
                newDataFrame = ratingsOfChosenBanditWithGenresithdata[
                    ratingsOfChosenBanditWithGenresithdata['Arm_K'] == arm].head(leastLongTimeHorizon)
                newDataFrame['Arm_K'] = k
                k = k + 1
                newDataFrame['Time_T'] = range(1, leastLongTimeHorizon + 1)
                newData.append(newDataFrame)

            ratingsOfChosenBanditWithGenresithdata = pd.concat(newData)

            if len(ratingsOfChosenBanditWithGenresithdata['Arm_K'].unique().tolist()) < K:
                K = K - 1
            else:
                break

        temp_T = np.minimum(leastLongTimeHorizon, int(T_set[iter_T]))
        T_allVariations.append(temp_T)

        regretUCB_T[iter_i, iter_T] = UCB1_data(ratingsOfChosenBanditWithGenresithdata, mu, K, temp_T) / temp_T
        regretMinimax_T[iter_i, iter_T] = \
            BASEFunc_data(ratingsOfChosenBanditWithGenresithdata, mu, K, temp_T, M, 'minimax', gamma) / temp_T
        regretGeometric_T[iter_i, iter_T] = \
            BASEFunc_data(ratingsOfChosenBanditWithGenresithdata, mu, K, temp_T, M, 'geometric', gamma) / temp_T
        regretArithmetic_T[iter_i, iter_T] = \
            BASEFunc_data(ratingsOfChosenBanditWithGenresithdata, mu, K, temp_T, M, 'arithmetic', gamma) / temp_T

regretMinimax_T_mean = np.mean(regretMinimax_T, axis=0)
regretGeometric_T_mean = np.mean(regretGeometric_T, axis=0)
regretArithmetic_T_mean = np.mean(regretArithmetic_T, axis=0)
regretUCB_T_mean = np.mean(regretUCB_T, axis=0)

# comparison with [PRCS16]

regretMinimax = np.zeros((m, len(M_set)), dtype=float)
regretGeometric = np.zeros((m, len(M_set)), dtype=float)
regretPRCSminimax = np.zeros((m, len(M_set)), dtype=float)
regretPRCSgeometric = np.zeros((m, len(M_set)), dtype=float)
regretUCB = np.zeros((m,), dtype=float)
mu = np.concatenate(([mu_max], [mu_min]), axis=0)

T_allVariations.clear()
for iter_i in range(0, m):
    print("Calcolando Diagramma 4, iter_i(da 0 a ", str(m - 1), ") =", str(iter_i)) 

    # Choosing the only the ratings of movies from the ith bandit
    ratingsOfChosenBanditWithGenresith = \
        ratingsOfChosenBanditWithGenres[ratingsOfChosenBanditWithGenres[
            'Bandit'].isin(ratingsOfChosenBanditWithGenres['Bandit'].value_counts()[iter_i:iter_i + 1].index.tolist())]

    leastLongTimeHorizon = 0
    ratingsOfChosenBanditWithGenresithdata = ratingsOfChosenBanditWithGenresith

    while leastLongTimeHorizon < 2:
        # Choosing the ratings on the k arms/genres most rated from the ith bandit
        ratingsOfChosenBanditWithGenresithdata = ratingsOfChosenBanditWithGenresith[
            ratingsOfChosenBanditWithGenresith[
                'Arm_K'].isin(ratingsOfChosenBanditWithGenresith['Arm_K'].value_counts()[:2].index.tolist())]

        leastLongTimeHorizon = ratingsOfChosenBanditWithGenresithdata['Arm_K'].value_counts().min()

        # Dropping di rating rows per genre/arm oltre il leastLongTimeHorizon
        # Renumbering of the arms from 1 to K
        # Numbering time horizin
        newData = []
        k = 1
        for arm in ratingsOfChosenBanditWithGenresithdata['Arm_K'].unique().tolist():
            newDataFrame = ratingsOfChosenBanditWithGenresithdata[
                ratingsOfChosenBanditWithGenresithdata['Arm_K'] == arm].head(leastLongTimeHorizon)
            newDataFrame['Arm_K'] = k
            k = k + 1
            newDataFrame['Time_T'] = range(1, leastLongTimeHorizon + 1)
            newData.append(newDataFrame)

        ratingsOfChosenBanditWithGenresithdata = pd.concat(newData)

    T = leastLongTimeHorizon
    T_allVariations.append(T)

    regretUCB[iter_i] = UCB1_data(ratingsOfChosenBanditWithGenresithdata, mu, 2, T)
    for iter_M in range(0, len(M_set)):
        temp_M = M_set[iter_M]
        regretMinimax[iter_i, iter_M] = \
            BASEFunc_data(ratingsOfChosenBanditWithGenresithdata, mu, 2, T, temp_M, 'minimax', gamma)
        regretGeometric[iter_i, iter_M] = \
            BASEFunc_data(ratingsOfChosenBanditWithGenresithdata, mu, 2, T, temp_M, 'geometric', gamma)
        regretPRCSminimax[iter_i, iter_M] = \
            PRCS_twoarm_data(ratingsOfChosenBanditWithGenresithdata, mu, temp_M, T, 'minimax')
        regretPRCSgeometric[iter_i, iter_M] = \
            PRCS_twoarm_data(ratingsOfChosenBanditWithGenresithdata, mu, temp_M, T, 'geometric')

T = np.mean(T_allVariations, axis=0)

regretMinimax_mean = np.mean(regretMinimax, axis=0) / T
regretGeometric_mean = np.mean(regretGeometric, axis=0) / T
regretPRCSminimax_mean = np.mean(regretPRCSminimax, axis=0) / T
regretPRCSGeometric_mean = np.mean(regretPRCSgeometric, axis=0) / T
regretUCB_mean = np.mean(regretUCB, axis=0) / T

# Figures

plt.figure(0)
plt.plot(M_set, regretMinimax_M_mean, marker='s', markerfacecolor='b',
         linestyle='-', color='b', linewidth=2)
plt.plot(M_set, regretGeometric_M_mean, marker='o', markerfacecolor='r',
         linestyle='--', color='r', linewidth=2)
plt.plot(M_set, regretArithmetic_M_mean, marker='v', markerfacecolor='c',
         linestyle='-.', color='c', linewidth=2)
plt.plot(M_set, regretUCB_M_mean * np.ones(len(M_set)), marker='.',
         markerfacecolor='k', linestyle=':', color='k', linewidth=2)
plt.xticks(range(2, 8))
plt.xlabel("$M$")
plt.ylabel('Average regret')
plt.legend(['Minimax Grid', 'Geometric Grid', 'Arithmetic Grid', 'UCB1'])

plt.figure(1)
plt.plot(K_set, regretMinimax_K_mean, marker='s', markerfacecolor='b',
         linestyle='-', color='b', linewidth=2)
plt.plot(K_set, regretGeometric_K_mean, marker='o', markerfacecolor='r',
         linestyle='--', color='r', linewidth=2)
plt.plot(K_set, regretArithmetic_K_mean, marker='v', markerfacecolor='c',
         linestyle='-.', color='c', linewidth=2)
plt.plot(K_set, regretUCB_K_mean, markerfacecolor='k',
         linestyle=':', color='k', linewidth=2)
plt.xticks(range(2, 19))
plt.xlabel("$K$")
plt.ylabel('Average regret')
plt.legend(['Minimax Grid', 'Geometric Grid', 'Arithmetic Grid', 'UCB1'])

plt.figure(2)
plt.plot(T_set, regretMinimax_T_mean, marker='s', markerfacecolor='b',
         linestyle='-', color='b', linewidth=2)
plt.plot(T_set, regretGeometric_T_mean, marker='o', markerfacecolor='r',
         linestyle='--', color='r', linewidth=2)
plt.plot(T_set, regretArithmetic_T_mean, marker='v', markerfacecolor='c',
         linestyle='-.', color='c', linewidth=2)
plt.plot(T_set, regretUCB_T_mean, markerfacecolor='k',
         linestyle=':', color='k', linewidth=2)
plt.xlim([int(T_first_min), int(T_first_max)])
plt.xscale('log')
plt.xlabel("$T$")
plt.ylabel('Average regret')
plt.legend(['Minimax Grid', 'Geometric Grid', 'Arithmetic Grid', 'UCB1'])

plt.figure(3)
plt.plot(M_set, regretMinimax_mean, marker='s', markerfacecolor='b',
         linestyle='-', color='b', linewidth=2)
plt.plot(M_set, regretGeometric_mean, marker='s', markerfacecolor='b',
         linestyle='--', color='b', linewidth=2)
plt.plot(M_set, regretPRCSminimax_mean, marker='o', markerfacecolor='r',
         linestyle='-', color='r', linewidth=2)
plt.plot(M_set, regretPRCSGeometric_mean, marker='o', markerfacecolor='r',
         linestyle='--', color='r', linewidth=2)
plt.plot(M_set, regretUCB_mean * np.ones(len(M_set)),
         markerfacecolor='k', linestyle=':', color='k', linewidth=2)
plt.xticks(range(2, 8))
plt.xlabel("$M$")
plt.ylabel('Average regret')
plt.legend(['BaSE: Minimax Grid', 'BaSE: Geometric Grid', 'ETC: Minimax Grid', 'ETC: Geometric Grid', 'UCB1'])

plt.show()
