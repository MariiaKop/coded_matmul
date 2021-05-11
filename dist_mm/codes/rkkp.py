import numpy as np
from scipy.linalg import khatri_rao


def encode_RKKP_matrix(matrix_parts, N, seed=None):
    parts = len(matrix_parts)

    np.random.seed(seed)
    R = np.random.rand(N, parts)

    matrix_tilda = [sum(matrix_parts[j] * r[j]  for j in range(parts)) for r in R]

    return matrix_tilda, R


def recovery_matrix_RKKP(C_tilda, saved_nodes, P, Q, l, p):
    (N, m), n = P.shape, Q.shape[1]

    G = khatri_rao(P.T, Q.T).T
    G_pinv = np.linalg.pinv(G[saved_nodes])

    C_tilda_reshaped = C_tilda.reshape(len(saved_nodes), -1)
    
    C_rec_reshaped = G_pinv @ C_tilda_reshaped

    C_rec_reshaped = np.vstack([C_i.reshape(l, p) for C_i in C_rec_reshaped])

    C_rec = np.hstack(np.split(C_rec_reshaped, m*n))
    C_rec = np.vstack(np.split(C_rec, m, 1))

    return C_rec


def recovery_matrix_RKKP_Strassen(C_tilda, saved_nodes, P, Q, l, p):
    (N, m), n = P.shape, Q.shape[1]

    G = khatri_rao(P.T, Q.T).T
    G_pinv = np.linalg.pinv(G[saved_nodes])

    C_tilda_reshaped = C_tilda.reshape(len(saved_nodes), -1)

    C_rec_reshaped = G_pinv @ C_tilda_reshaped

    C_rec = C_rec_reshaped[::m+1]
    C_rec = np.vstack([np.split(c, l) for c in C_rec])

    return C_rec
