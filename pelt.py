import numpy as np


def pelt(data, cost, penalty=None):
    '''

    Compute changepoints in time series using PELT algo

    Reference:
        Killick R, Fearnhead P, Eckley IA (2012) Optimal detection
            of changepoints with a linear computational cost, JASA
            107(500), 1590-1598

     param: data: array
         visits per minute order by ams_time
     param: cost: function  (int, int) -> float
     param: penalty: float, optional, default: log(len(data))
     returns: list
         Indexes of changepoints
    '''
    n = len(data)

    if penalty is None:
        penalty = np.log(n)

    F = np.zeros(n + 1)
    F[0] = -penalty
    F[1] = 0

    R = np.array([0])
    chpts = np.zeros(n)

    for t in range(1, n):
        cpt_cands = R
        seg_costs = np.zeros(len(cpt_cands))
        for i in range(0, len(cpt_cands)):
            seg_costs[i] = cost(cpt_cands[i], t)

        tt = [(F[i]) for i in cpt_cands]
        temp = tt + seg_costs + penalty
        tau = np.argmin(temp)
        F[t] = temp[tau]
        chpts[t] = cpt_cands[tau]

        # pruning step
        ineq_prune = [el < F[t] for el in tt + seg_costs]
        tr = [cpt_cands[i] for i, v in enumerate(ineq_prune) if v]
        R = tr + [t - 1]

    # get changepoints
    last = int(chpts[-1])
    CP = [last]
    while last > 0:
        last = chpts[int(last)]
        CP.append(int(last))

    CP.sort()

    return chpts
