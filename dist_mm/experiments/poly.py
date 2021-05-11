import sys
from time import time

import numpy as np
from mpi4py import MPI

from dist_mm.tools import (split_matrix, choice_straggler_nodes,
                           load_config, multiply_parts, save_logs)
from dist_mm.codes.poly import encode_poly_matrix, recovery_matrix_poly


comm = MPI.COMM_WORLD
N = comm.Get_size() - 1
RANK = comm.Get_rank()


cfg = load_config(sys.argv[1], RANK, comm)
if RANK == 0:
    print('Parameters:', cfg, end='\n\n')
m = cfg['m']
n = cfg['n']
s = cfg['s']
r = cfg['r']
t = cfg['t']
stragglers = cfg['stragglers']
random_state = cfg['random_state']

l, p = r // m, t // n


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
    A_parts = split_matrix(A, m)
    B_parts = split_matrix(B, n)

    A_tilda = encode_poly_matrix(A_parts, N, 1)
    B_tilda = encode_poly_matrix(B_parts, N, m)

    encoded_time = time()
    logging['encoding_time'] = encoded_time - start_time

    C_tilda = [np.empty((l, p), dtype='d') for i in range(N)]
    req_A = [comm.Isend([A_tilda[i], MPI.DOUBLE], dest=i+1, tag=11) for i in range(N)]
    req_B = [comm.Isend([B_tilda[i], MPI.DOUBLE], dest=i+1, tag=22) for i in range(N)]
    req_C = [comm.Irecv([C_tilda[i], MPI.DOUBLE], source=i+1, tag=33) for i in range(N)]

    MPI.Request.Waitall(req_A)
    MPI.Request.Waitall(req_B)

    sent_time = time()
    logging['sending_time'] = sent_time - encoded_time

    saved_nodes = np.array([MPI.Request.Waitany(req_C) for _ in range(recovery_th)])
    received_time = time()

    print('The fastest nodes:', saved_nodes)
    logging['fastest_nodes'] = saved_nodes.tolist()

    saved_nodes = sorted(saved_nodes)
    C_tilda = np.vstack([C_tilda[i] for i in saved_nodes])
    logging['receiving_time'] = received_time - sent_time

    start_decoding_time = time()
    C_rec = recovery_matrix_poly(C_tilda, N, m, n, saved_nodes, l, p)
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
    multiply_parts(comm, s, l, p, 1)
