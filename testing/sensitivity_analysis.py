"""

"""

import numpy as np
from numpy import linalg as la
from icecream import ic

def perturb_states(s, magnitude):
    """
    Perturbs true sequence of states based on magnitude of pertubation.
    """
    raise NotImplementedError

def F(s):
    """
    Transition function, normally would be neural network. Currently hard coded to match example.
    """
    v_0 = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    
    v_1 = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    
    F_v_0 = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0]
    ])
    
    F_v_1 = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])
    
    if np.all(s == v_0):
        return F_v_0
    elif np.all(s == v_1):
        return F_v_1
    elif not np.any(s):  # if s is the zero matrix
        return np.zeros((3, 3), dtype=np.int32)
    else:
        raise NotImplementedError

def d(s_1, s_2):
    """
    Difference operator
    """
    assert s_1[0] == s_2[0], f's_1[0] must equal s_2[0]; however, s_1[0] = {s_1[0]} != s_2[0] = {s_2[0]}'
    assert len(s_1) == len(s_2), f's_1 must be the same size as s_2; however, |s_1| = {len(s_1)} != |s_2| = {len(s_2)}'
    return sum([2 ** -(t + 1) * la.norm(s_1[t] - s_2[t]) ** 2 for t in range(len(s_1))])

def e(s_1, s_2):
    """
    Error operator for state matrices at time t.
    
    :returns: xor of states
    :rtype: np.ndarray
    """
    return s_1 ^ s_2

def d_t(s):
    """
    Makes d matrix for the construction of A block matrix.
    """
    s_copy = s.copy()
    value_to_indices = {v: set() for v in {-1, 0, 1}}
    
    value_conversion = {  # TODO: figure out what these are supposed to be
        -1: 0,
        0: 0,
        1: -1
    }
    
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            value_to_indices[s[i, j]].add((i, j))

    for value, indices in value_to_indices.items():
        for index in indices:
            s_copy[index] = value_conversion[value]
    
    return s_copy

def main():
    """
    OBJECTIVES:
        1) Assess the robustness of true sequence of states 'u'
    METHOD:
        1) Produce pseudo orbit 'v' such that in a metric 'd' we satisfy 'd(v_{t+1}, F(v_t)) <= lambda_t for lambda_t <= lambda' for potentially varying 'lambda'
        2) Using pseudo orbit 'v' as initial guess, solve optimization problem 'min_w H(w)' where 'H(w) = sum(d(w_{t+1}, F(w_t))'
        3) Collect statistics on convergence of optimized 'w' as function of lambda
    """
    
    # initialize states (currently hard-coded)
    m, n = 3, 3  # size of state at time t (size of game map)
    l = 1  # cardinality of states (length of snake)
    T = 3  # total number of time steps
    #perturbation_magnitude = 1
    
    true_states = u = np.array([  # true states (actual snake)
        [  # t = 0
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ],
        [  # t = 1
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [  # t = 2
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0]
        ]
    ])
    
    #perturbed_states = perturb_states(true_states, perturbation_magnitude)
    perturbed_states = v = np.array([  # perturbed states
        [  # t = 0
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0]
        ],
        [  # t = 1
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [  # t = 2
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0]
        ]
    ])
    
    w = v.copy()

    # form block matrix E by evaluating 'e(w_t, w_{t + 1}) for t in {0, 1, ..., T - 1}'
    E = [e(w[t + 1], F(w[t])) for t in range(T - 1)]  # E has shape (T, m, n)
    
    
    # collect indices of active variables
    active_variables = [{(i, j) for i in range(m) for j in range(n) if E[t][i, j]} for t in range(T - 1)]
    ic(active_variables[0], active_variables[1])
    print()
    
    # form block matrix A by forming 'A(t, t) for t in {0, 1, ..., T - 1}'
    A = [np.zeros((m * n, m * n), dtype=np.int32) for t in range(T - 1)]  # initialize A with zeros
    
    for t in range(T - 1):  # for each time step t
        for j, k in active_variables[t]:  # for each active variable at time t
            column_index = j + 3 * k
            lhs = e(w[t + 1], F(w[t] + d_t(w[t])))
            rhs = E[t]
            A_t = (lhs - rhs).T.reshape(m * n)  # form block component and unroll result
            A[t][:, column_index] = A_t  # set the (j * k)th column of A
    
    ic(A[0])
    print()
    ic(A[1])
    return
        
    # ???
    
    # with full dimensionality
    M = np.array([  # form M by concatenating blocks
        [A[0], np.eye(m * n, m * n), np.zeros((m * n, m * n))],
        [np.zeros((m * n, m * n)), A[1], np.eye(m * n, m * n)]
    ])
    
    delta = -la.pinv(M) @ E
    
    # reduce dimensionality
    # ...

if __name__ == '__main__':
    main()