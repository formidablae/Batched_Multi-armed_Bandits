import ctypes
import math
import numpy as np


# function of BASE
# parameters
#     K: number of batches
#     TSeq: horizon
#     M: number of batches
#     TGridMinimax: minimax batch grids
#     mu: batch mean
#     gamma: tunning parameter


def BASEFunc(mu, K, T, M, gridType, gamma) -> float:
    # record
    regret = 0
    if gridType == 'minimax':
        a = T ** (1 / (2 - 2**(1 - M)))
        TGrid = np.floor(np.power(
            a, np.subtract(2, np.divide(1, np.power(2, (range(0, M)))))))
        TGrid[M - 1] = T
        # minimax batch
        TGrid = np.concatenate(([0], TGrid), axis=0)
    elif gridType == 'geometric':
        b = T ** (1 / M)
        TGrid = np.floor(np.power(b, range(1, M+1)))
        TGrid[M - 1] = T
        # adaptive batch grids
        TGrid = np.concatenate(([0], TGrid), axis=0)
    else:
        TGrid = np.floor(np.linspace(0, T, M + 1))

    # initialization
    activeSet = np.ones((K, 1), dtype=int)
    numberPull = np.zeros(K, dtype=int)
    averageReward = np.zeros(K, dtype=float)

    for i in range(1, M + 1):
        availableK = np.sum(activeSet)
        pullNumber = \
            int(np.round(np.maximum(np.floor((TGrid[i] -
                                              TGrid[i - 1]) / availableK), 1)))
        TGrid[i] = availableK * pullNumber + TGrid[i - 1]
        rowActiveSetOne, colActiveSetOne = np.where(activeSet == 1)
        for j in rowActiveSetOne:
            averageReward[j] = averageReward[j] * (
                    numberPull[j] / (numberPull[j] + pullNumber)) + (
                    np.mean(np.random.randn(1, pullNumber)) + mu[j]) * (
                    pullNumber / (numberPull[j] + pullNumber))
            regret = regret + (pullNumber * (mu[0] - mu[j]))
            numberPull[j] = numberPull[j] + pullNumber

        rowActiveSetOne, colActiveSetOne = np.where(activeSet == 1)
        maxArm = np.max(averageReward[rowActiveSetOne])
        for j in rowActiveSetOne:
            if (maxArm - averageReward[j]) >= np.sqrt(
                    gamma * math.log(T * K) / numberPull[j]):
                activeSet[j] = 0

    return regret
