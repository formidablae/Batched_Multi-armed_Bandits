import math
import numpy as np

from matplotlib import pyplot as plt

from UCB1 import UCB1
from BASEFunc import BASEFunc
from PRCS_twoarm import PRCS_twoarm

"""
Parametri
"""

# K = 3 
# T = 5*10^4
# M = 3
K = 3
T = int(5e4)
M = 3

# K_set = vettore riga di 6 punti da log10(2) a log10(20) ugualmente
# logaritmicamente distanti (arrotondati)
# T_set = vettore riga di 6 punti da log10(500) a log10(5*10^4) ugualmente
# logaritmicamente distanti (arrotondati)
# M_set = insieme dei punti da 2 a 7 (compreso).
K_set = np.round(np.logspace(math.log(2, 10), math.log(20, 10), num=6,
                               endpoint=True,
                               base=10.0,
                               dtype=None, axis=0))
T_set = np.round(np.logspace(math.log(500, 10), math.log(int(5e4), 10), num=6,
                               endpoint=True,
                               base=10.0,
                               dtype=None, axis=0))
M_set = range(2, 8)

# mu_max = 0.6 reward medio per l'arm ottimale
# mu_min = 0.5 reward medio per tutti gli altri arm
# gamma = 0.5
# m = 200
mu_max = 0.6
mu_min = 0.5
gamma = 0.5
m = 200

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

for iter_i in range(0, m):
    print("Calculating First  "+str(iter_i))
    regretUCB_M[iter_i] = UCB1(mu, K, T)
    for iter_M in range(0, len(M_set)):
        temp_M = M_set[iter_M]
        regretMinimax_M[iter_i, iter_M] = \
            BASEFunc(mu, K, T, temp_M, 'minimax', gamma)
        regretGeometric_M[iter_i, iter_M] = \
            BASEFunc(mu, K, T, temp_M, 'geometric', gamma)
        regretArithmetic_M[iter_i, iter_M] = \
            BASEFunc(mu, K, T, temp_M, 'arithmetic', gamma)

regretMinimax_M_mean = np.mean(regretMinimax_M, axis=0) / T
regretGeometric_M_mean = np.mean(regretGeometric_M, axis=0) / T
regretArithmetic_M_mean = np.mean(regretArithmetic_M, axis=0) / T
regretUCB_M_mean = np.mean(regretUCB_M, axis=0) / T


# dependence on K

regretMinimax_K = np.zeros((m, len(K_set)), dtype=float)
regretGeometric_K = np.zeros((m, len(K_set)), dtype=float)
regretArithmetic_K = np.zeros((m, len(K_set)), dtype=float)
regretUCB_K = np.zeros((m, len(K_set)), dtype=float)

for iter_i in range(0, m):
    print("Calculating Second "+str(iter_i))
    for iter_K in range(0, len(K_set)):
        temp_K = int(K_set[iter_K])
        mu = np.concatenate(([mu_max], mu_min *
                             np.ones((1, temp_K-1),
                                     dtype=int).flatten()), axis=0)
        regretUCB_K[iter_i, iter_K] = UCB1(mu, temp_K, T)
        regretMinimax_K[iter_i, iter_K] = \
            BASEFunc(mu, temp_K, T, M, 'minimax', gamma)
        regretGeometric_K[iter_i, iter_K] = \
            BASEFunc(mu, temp_K, T, M, 'geometric', gamma)
        regretArithmetic_K[iter_i, iter_K] = \
            BASEFunc(mu, temp_K, T, M, 'arithmetic', gamma)


regretMinimax_K_mean = np.mean(regretMinimax_K, axis=0) / T
regretGeometric_K_mean = np.mean(regretGeometric_K, axis=0) / T
regretArithmetic_K_mean = np.mean(regretArithmetic_K, axis=0) / T
regretUCB_K_mean = np.mean(regretUCB_K, axis=0) / T


# dependence on T

regretMinimax_T = np.zeros((m, len(T_set)), dtype=float)
regretGeometric_T = np.zeros((m, len(T_set)), dtype=float)
regretArithmetic_T = np.zeros((m, len(T_set)), dtype=float)
regretUCB_T = np.zeros((m, len(T_set)), dtype=float)
mu = np.concatenate(([mu_max], mu_min * np.ones((1, K-1),
                                                dtype=int).flatten()), axis=0)

for iter_i in range(0, m):
    print("Calculating Third  "+str(iter_i))
    for iter_T in range(0, len(T_set)):
        temp_T = int(T_set[iter_T])
        regretUCB_T[iter_i, iter_T] = UCB1(mu, K, temp_T) / temp_T
        regretMinimax_T[iter_i, iter_T] = \
            BASEFunc(mu, K, temp_T, M, 'minimax', gamma) / temp_T
        regretGeometric_T[iter_i, iter_T] = \
            BASEFunc(mu, K, temp_T, M, 'geometric', gamma) / temp_T
        regretArithmetic_T[iter_i, iter_T] = \
            BASEFunc(mu, K, temp_T, M, 'arithmetic', gamma) / temp_T


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

for iter_i in range(0, m):
    print("Calculating Fourth "+str(iter_i))
    regretUCB[iter_i] = UCB1(mu, 2, T)
    for iter_M in range(0, len(M_set)):
        temp_M = M_set[iter_M]
        regretMinimax[iter_i, iter_M] = \
            BASEFunc(mu, 2, T, temp_M, 'minimax', gamma)
        regretGeometric[iter_i, iter_M] = \
            BASEFunc(mu, 2, T, temp_M, 'geometric', gamma)
        regretPRCSminimax[iter_i, iter_M] = \
            PRCS_twoarm(mu, temp_M, T, 'minimax')
        regretPRCSgeometric[iter_i, iter_M] = \
            PRCS_twoarm(mu, temp_M, T, 'geometric')

regretMinimax_mean = np.mean(regretMinimax, axis=0) / T
regretGeometric_mean = np.mean(regretGeometric, axis=0) / T
regretPRCSminimax_mean = np.mean(regretPRCSminimax, axis=0) / T
print("regretPRCSgeometric = \n"+str(regretPRCSgeometric))
regretPRCSGeometric_mean = np.mean(regretPRCSgeometric, axis=0) / T
print("regretPRCSGeometric_mean = \n"+str(regretPRCSGeometric_mean))
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
plt.xticks(range(2, 21))
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
plt.xlim([int(5e2), int(5e4)])
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
