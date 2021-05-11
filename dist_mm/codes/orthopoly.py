import numpy as np
import numpy.polynomial.chebyshev as cheb


def T(idx=0):
    coefs = np.zeros(idx+1)
    coefs[idx] = 1

    return cheb.Chebyshev(coefs)


def generate_matrix_H(m, n):
    def generate_submatrix_H(m):
        H_i = np.zeros((2*m - 1, m))
        diagonal_part = np.diag(np.ones(m-1)/2)
        H_i[m-1, 0] = 1
        H_i[-m+1:, -m+1:] = diagonal_part
        H_i[:m-1, -m+1:] = np.rot90(diagonal_part)

        return H_i

    H = np.zeros((m*n, m*n)) 
    H_i = generate_submatrix_H(m)

    H[:m, :m] = np.eye(m)

    for i in range(1, n):
        H[m*(i-1) + 1: m*(i+1), i*m : (i+1)*m] = H_i

    return H


def encode_cheb_matrix(matrix_parts, N, m):
    x = np.arange(1, N + 1)
    x = np.cos((2*x - 1) * np.pi / 2 / N)

    parts = len(matrix_parts)
    matrix_tilda = [sum(matrix_parts[j] * T(j*m)(xi) for j in range(parts)) for xi in x]

    return matrix_tilda


def recovery_matrix_cheb(C_tilda, N, m, n, saved_nodes, l, p):
    x = np.arange(1, N + 1)
    x = np.cos((2*x - 1) * np.pi / 2 / N)

    chebvander = cheb.chebvander(x, m*n - 1).astype(np.float64)

    H = generate_matrix_H(m, n)

    chebvander_pinv = np.linalg.pinv((chebvander @ H) [saved_nodes])

    C_tilda_reshaped = np.hstack([C_tilda[i::l] for i in range(l)])
    C_rec_reshaped = chebvander_pinv @ C_tilda_reshaped

    C_rec = np.vstack([C_i.reshape(l, p) for C_i in C_rec_reshaped])
    C_rec = np.hstack(np.split(C_rec, m))

    return C_rec


def recovery_matrix_cheb_Strassen(C_tilda, N, m, n, saved_nodes, l, p):
    x = np.arange(1, N + 1)
    x = np.cos((2*x - 1) * np.pi / 2 / N)

    chebvander = cheb.chebvander(x, m*n - 1).astype(np.float64)

    H = generate_matrix_H(m, n)

    chebvander_pinv = np.linalg.pinv((chebvander @ H) [saved_nodes])

    C_tilda_reshaped = np.hstack([C_tilda[i::l] for i in range(l)])
    C_rec_reshaped = chebvander_pinv @ C_tilda_reshaped

    C_rec = C_rec_reshaped[::m+1]
    C_rec = np.vstack([np.split(c, l) for c in C_rec])

    return C_rec
