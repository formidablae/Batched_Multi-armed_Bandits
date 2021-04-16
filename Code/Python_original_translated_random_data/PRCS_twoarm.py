import math
import numpy as np


def PRCS_twoarm(mu, M, T, gridType) -> float:
    if gridType == 'minimax':
        a = T**(1 / (2 - 2**(1-M)))
        TGrid = np.floor(np.power(a, np.subtract(2, np.divide(
            1, np.power(2, (range(0, M)))))))
        TGrid[M-1] = T
        # minimax batch grids
        TGrid = np.concatenate(([0], TGrid), axis=0)
    elif gridType == 'geometric':
        b = T**(1 / M)
        TGrid = np.floor(np.power(b, range(1, M+1)))
        TGrid[M-1] = T
        # adaptive batch grids
        TGrid = np.concatenate(([0], TGrid), axis=0)

    pullnumber = int(np.round(TGrid[1] / 2))
    regret = pullnumber * (mu[0] - mu[1])
    reward = np.sum(np.concatenate((
        [np.add(np.random.randn(pullnumber,), mu[0])],
        [np.add(np.random.randn(pullnumber,), mu[1])]), axis=0), axis=1)
    opt = 0

    for m in range(1, M):
        t = TGrid[m]
        thres = np.sqrt(4 * math.log(2 * T / t) / t)
        if opt == 0:
            if (reward[0] - reward[1]) / t > thres:
                opt = 1
            elif (reward[1] - reward[0]) / t > thres:
                opt = 2
            else:
                cur_number = int(np.round((TGrid[m+1] - TGrid[m]) / 2))
                pullnumber = pullnumber + cur_number
                reward = reward + np.sum(np.concatenate((
                    [np.add(np.random.randn(cur_number,), mu[0])],
                    [np.add(np.random.randn(cur_number,), mu[1])]),
                    axis=0), axis=1)
                regret = regret + cur_number * (mu[0] - mu[1])

        if opt == 2:
            regret = regret + (TGrid[m + 1] - TGrid[m]) * (mu[0] - mu[1])

        if m == (M-2):
            if reward[0] > reward[1]:
                opt = 1
            else:
                opt = 2

    return regret
