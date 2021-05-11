import numpy as np


def encode_poly_matrix(matrix_parts, N, power=1):
    x = np.arange(1, N + 1)
    x = np.linspace(-1, 1, N)
    parts = len(matrix_parts)

    matrix_tilda = [sum(matrix_parts[j] * xi**(j*power) for j in range(parts)) for xi in x]

    return matrix_tilda


def recovery_matrix_poly(C_tilda, N, m, n, saved_nodes, l, p):
    x = np.linspace(-1, 1, N)
    vander = np.vander(x, m*n, increasing=True).astype(np.float64)
    vander_pinv = np.linalg.pinv(vander[saved_nodes])

    C_tilda_reshaped = np.hstack([C_tilda[i::l] for i in range(l)])
    C_rec_reshaped = vander_pinv @ C_tilda_reshaped

    C_rec = np.vstack([C_i.reshape(l, p) for C_i in C_rec_reshaped])
    C_rec = np.hstack(np.split(C_rec, m))

    return C_rec


def recovery_matrix_poly_Strassen(C_tilda, N, m, n, saved_nodes, l, p):
    x = np.linspace(-1, 1, N)
    vander = np.vander(x, m*n, increasing=True).astype(np.float64)
    vander_pinv = np.linalg.pinv(vander[saved_nodes])

    C_tilda_reshaped = np.hstack([C_tilda[i::l] for i in range(l)])
    C_rec_reshaped = vander_pinv @ C_tilda_reshaped

    C_rec = C_rec_reshaped[::m+1]
    C_rec = np.vstack([np.split(c, l) for c in C_rec])

    return C_rec
