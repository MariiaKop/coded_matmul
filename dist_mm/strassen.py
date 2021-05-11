import numpy as np


def divide_matrix(X):
    n, m = X.shape

    assert n > 1
    assert m > 1

    n //= 2
    m //= 2

    X_parts = [X[:n, :m], X[:n, m:], X[n:, :m], X[n:, m:]]
    return X_parts


def Strassen_division(A, B):
    A1, A3, A2, A4 = divide_matrix(A)
    B1, B2, B3, B4 = divide_matrix(B)

    S = [A1+A4, A3+A4, A1, A4, A1+A2, A3-A1, A2-A4]
    Q = [B1+B4, B1, B2-B4, B3-B1, B4, B1+B2, B3+B4]

    return S, Q


def collect_Strassen_result(P):
    C_parts = [P[0] + P[3] + P[6] - P[4],
               P[2] + P[4], P[1] + P[3],
               P[0] + P[2] + P[5] - P[1]
              ]

    C_rec = np.vstack((np.hstack(C_parts[:2]), np.hstack(C_parts[2:])))

    return C_rec
