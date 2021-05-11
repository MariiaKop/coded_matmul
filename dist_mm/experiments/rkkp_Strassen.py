import sys
from time import time

import numpy as np
from mpi4py import MPI

from dist_mm.tools import (split_matrix, choice_straggler_nodes,
                           load_config, multiply_parts, save_logs)
from dist_mm.codes.rkkp import encode_RKKP_matrix, recovery_matrix_RKKP_Strassen
from dist_mm.strassen import Strassen_division, collect_Strassen_result


comm = MPI.COMM_WORLD
N = comm.Get_size() - 1
RANK = comm.Get_rank()


cfg = load_config(sys.argv[1], RANK, comm)
if RANK == 0:
    print('Parameters:', cfg, end='\n\n')
m = 7
n = 7
s = cfg['s']
r = cfg['r']
t = cfg['t']
stragglers = cfg['stragglers']
random_state = cfg['random_state']

l, p = r // 2, t // 2


if RANK == 0:
    logging = cfg
    logging['N'] = N
    np.random.seed(random_state)
    A = np.random.rand(s, r).astype('d')
    B = np.random.rand(s, t).astype('d')

    recovery_th = m*n

    straggler_nodes = choice_straggler_nodes(N, stragglers)
    logging['straggler_nodes'] = straggler_nodes.tolist()
    print('Straggler nodes:', straggler_nodes)

    for i in range(N):
        if i in straggler_nodes:
            comm.send(1, dest=i+1, tag=99)
        else:
            comm.send(0, dest=i+1, tag=99)

    start_time = time()
    S, Q = Strassen_division(A, B)

    S_tilda, U = encode_RKKP_matrix(S, N, random_state)
    Q_tilda, V = encode_RKKP_matrix(Q, N, random_state)

    encoded_time = time()
    logging['encoding_time'] = encoded_time - start_time

    P_tilda = [np.empty((l, p), dtype='d') for i in range(N)]
    req_A = [comm.Isend([S_tilda[i], MPI.DOUBLE], dest=i+1, tag=11) for i in range(N)]
    req_B = [comm.Isend([Q_tilda[i], MPI.DOUBLE], dest=i+1, tag=22) for i in range(N)]
    req_C = [comm.Irecv([P_tilda[i], MPI.DOUBLE], source=i+1, tag=33) for i in range(N)]

    MPI.Request.Waitall(req_A)
    MPI.Request.Waitall(req_B)

    sent_time = time()
    logging['sending_time'] = sent_time - encoded_time

    saved_nodes = np.array([MPI.Request.Waitany(req_C) for _ in range(recovery_th)])
    received_time = time()

    print('The fastest nodes:', saved_nodes)
    logging['fastest_nodes'] = saved_nodes.tolist()

    saved_nodes = sorted(saved_nodes)
    P_tilda = np.vstack([P_tilda[i] for i in saved_nodes])
    logging['receiving_time'] = received_time - sent_time

    start_decoding_time = time()
    P_rec = recovery_matrix_RKKP_Strassen(P_tilda, saved_nodes, U, V, l, p)
    C_rec = collect_Strassen_result(np.split(P_rec, m))

    end_time = time()
    logging['decoding_time'] = end_time - start_decoding_time
    logging['time'] = end_time - start_time

    print(f'Sending time: {logging["sending_time"]:.2f} sec')
    print(f'Receiving time: {logging["receiving_time"]:.2f} sec', end='\n\n')

    print(f'Encoding time: {logging["encoding_time"]:.2f} sec')
    print(f'Decoding time: {logging["decoding_time"]:.2f} sec')
    print('Encoding + decoding time:',
          f'{logging["encoding_time"]+logging["decoding_time"]:.2f} sec', end='\n\n')

    print(f'Time: {logging["time"]:.2f} sec')
    print(f'Ratio: {(logging["encoding_time"]+logging["decoding_time"]) /logging["time"]:.2f}')

    C = A.T @ B
    nmse = np.linalg.norm(abs(C - C_rec))/np.linalg.norm(C)
    print('NMSE:', nmse)
    logging['nmse'] = nmse

    if len(sys.argv) == 3:
        save_logs(logging, sys.argv[2])

else:
    multiply_parts(comm, s, l, p, timeout=1)
