from time import time

import yaml
import json
import numpy as np


def split_matrix(matrix, parts):
    '''Genearating submatrices'''
    assert parts <= matrix.shape[1]

    return np.split(matrix, parts, axis=1)


def stragle_random_part(A_tilda, B_tilda, straggers=0, seed=None):
    if straggers == 0:
        return A_tilda, B_tilda, np.arange(len(A_tilda))

    assert len(A_tilda) == len(B_tilda)

    saved_nodes = choice_saved_nodes(len(A_tilda), straggers, seed)

    A_tilda = [A_tilda[i] for i in saved_nodes]
    B_tilda = [B_tilda[i] for i in saved_nodes]

    return A_tilda, B_tilda, saved_nodes


def choice_saved_nodes(N, straggers, seed=None):
    if straggers == 0:
        return np.arange(N)

    np.random.seed(seed)
    saved_nodes = np.random.choice(range(N), size=N-straggers, replace=False)
    saved_nodes = sorted(saved_nodes)

    return saved_nodes


def choice_straggler_nodes(N, straglers, seed=None):
    if straglers == 0:
        return np.array([])

    np.random.seed(seed)
    straggler_nodes = np.random.choice(range(N), size=straglers, replace=False)

    return straggler_nodes


def loop(timeout):
    t = time()
    a = 0
    while time() < t + timeout:
        a += 1


def load_config(path_to_config, rank, comm):
    if rank == 0:
        with open(path_to_config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = None

    cfg = comm.bcast(cfg, root=0)
    return cfg


def multiply_parts(comm, s, l, p, timeout=1):
    straggler = comm.recv(source=0, tag=99)

    A_node = np.empty((s, l), dtype='d')
    B_node = np.empty((s, p), dtype='d')

    req_A = comm.Irecv(A_node, source=0, tag=11)
    req_B = comm.Irecv(B_node, source=0, tag=22)

    req_A.wait()
    req_B.wait()

    if straggler:
        loop(timeout)

    C_node = A_node.T @ B_node
    req_C = comm.Isend(C_node, dest=0, tag=33)
    req_C.Wait()


def save_logs(logging, path):
    with open(path, 'w') as f:
        json.dump(logging, f)
